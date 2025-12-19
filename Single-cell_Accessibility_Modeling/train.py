import sys
import os
import time
import configargparse
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils import fix_random, load_data, ModelEma, EarlyStopping

def make_parser():
    parser = configargparse.ArgParser(description="")
    parser.add_argument('--data', type=str, default='../processed_data')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=10)
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()
    '''================================================================'''
    device_id = args.gpu
    output_path = os.path.join(args.data, 'train_output')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % device_id[0] if use_cuda else "cpu")
    fix_random(args.seed)
    torch.cuda.empty_cache()

    tra_set, val_set, num_cell = load_data(data_path=args.data)
    num_worker = int(len(device_id) * 4)
    tra_DataLoader = torch.utils.data.DataLoader(tra_set, batch_size=args.batch_size, shuffle=True, drop_last=True,num_workers=num_worker, pin_memory=True, )
    val_DataLoader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, num_workers=num_worker,pin_memory=True)

    from Model import model
    model = model(num_cell=num_cell)
    model = nn.DataParallel(model, device_ids=device_id)
    model.to(device)
    ema = ModelEma(model, 0.9997)
    scaler = torch.cuda.amp.GradScaler()

    loss_fun = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.95, 0.9995), weight_decay=1e-3)
    scheduler = None

    if not os.path.exists(output_path): os.mkdir(output_path)
    writer = SummaryWriter(log_dir=output_path)
    early_stopping_val_auc = EarlyStopping(patience=30, verbose=True)
    best_auc = 0

    total_iters = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        torch.cuda.empty_cache()
        '''======================================================================='''
        model.train()
        total_tra_loss = 0
        data_loader = tqdm(tra_DataLoader, file=sys.stdout)
        for tra_id, (tra_x, tra_y) in enumerate(data_loader):
            # sig = nn.Sigmoid()
            optimizer.zero_grad()
            tra_x = tra_x.to(device, dtype=torch.float32)
            tra_y = tra_y.to(device, dtype=torch.float32)

            with torch.cuda.amp.autocast():
                tra_output = model(tra_x)
                # tra_output_sig = sig(tra_output)
                loss = loss_fun(tra_output, tra_y)
            if torch.isnan(loss):
                print(f"Step {tra_id} loss is NaN. Saving gradients for inspection.")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                total_iters += 1
                scheduler.step()
                curr_lr = scheduler.get_lr()
                writer.add_scalar('lr', curr_lr[0], total_iters)
            else:
                curr_lr = [args.lr]
            total_tra_loss += loss.item()
            ema.update(model)
        epoch_train_loss = total_tra_loss / (tra_id + 1)
        '''=================================================================='''
        model.eval()
        total_val_loss = 0
        true, pred, pred_ema, auc, ema_auc = [], [], [], [], [],
        with torch.no_grad():
            val_bar = tqdm(val_DataLoader, file=sys.stdout)
            for val_id, (val_x, val_y) in enumerate(val_bar):
                sig = nn.Sigmoid()
                val_x = val_x.to(device, non_blocking=True, dtype=torch.float32)
                val_y = val_y.to(device, non_blocking=True, dtype=torch.float32)
                val_output = model(val_x)
                loss = loss_fun(val_output, val_y)
                total_val_loss += loss.item()
                new_out = sig(val_output)
                val_output_ema = sig(ema.module(val_x))
                true.append(val_y.cpu())
                pred.append(new_out.cpu())
                pred_ema.append(val_output_ema.cpu())
            true = torch.cat(true, 0).numpy()
            pred = torch.cat(pred, 0).numpy()
            pred_ema = torch.cat(pred_ema, 0).numpy()
            for i in range(true.shape[1]):
                auc.append(roc_auc_score(y_true=true[:, i], y_score=pred[:, i]))
                ema_auc.append(roc_auc_score(y_true=true[:, i], y_score=pred_ema[:, i]))
            epoch_val_loss = total_val_loss / (val_id + 1)
            epoch_val_auc = sum(auc) / len(auc)
            epoch_val_auc_ema = sum(ema_auc) / len(ema_auc)
        '''============================================================'''
        writer.add_scalars('loss', {'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}, epoch + 1)
        writer.add_scalars('AUC', {'val': epoch_val_auc, 'val_ema': epoch_val_auc_ema}, epoch + 1)

        t1 = time.time()
        print('[Epoch %d]|train_loss: %.5f|val_loss: %.5f|val_auc: %5f|val_auc_ema: %5f|Time used: %.2fs' % (
            epoch + 1, epoch_train_loss, epoch_val_loss, epoch_val_auc, epoch_val_auc_ema, t1 - t0,), flush=True)

        if max(epoch_val_auc, epoch_val_auc_ema) > best_auc:
            if epoch_val_auc_ema > epoch_val_auc:
                state_dict = ema.module.state_dict()
            else:
                state_dict = model.state_dict()
            opti_state = optimizer.state_dict()
            best_auc = max(epoch_val_auc, epoch_val_auc_ema)
            checkpoint = {'best_model_state': state_dict,
                          'best_optimizer_state': opti_state,
                          'current_train_loss': epoch_train_loss,
                          'current_val_loss': epoch_val_loss,
                          'val_auc': best_auc,
                          'epoch': epoch + 1}
            torch.save(checkpoint, '%s/best_val_auc_model.pth' % output_path)
        early_stopping_val_auc(metrics=max(epoch_val_auc, epoch_val_auc_ema))
        if early_stopping_val_auc.early_stop:
            print("Early stopping")
            break
    writer.close()




if __name__ == "__main__":
    main()





