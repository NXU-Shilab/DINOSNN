require(BSgenome.Hsapiens.UCSC.hg38)
require(rtracklayer)
require(TxDb.Hsapiens.UCSC.hg38.knownGene)
library(VariantAnnotation)
library(data.table)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: Rscript script.R <input_file> <output_file>")
}

in_file  <- args[1]
out_file <- args[2]

process.simple.offsets.python <- function(DF,
                                          window = 2788,
                                          filterUnique = FALSE,
                                          filterCoding = TRUE,
                                          out_file) {
  colnames(DF)[1:8] <- c("C","D","E","A","B","F","G","H")
  DF <- DF[!is.na(DF$D), ]
  DF$D <- as.numeric(as.character(as.matrix(DF$D)))
  halfw <- window / 2
  DF$C <- as.character(DF$C)
  chrends <- seqlengths(BSgenome.Hsapiens.UCSC.hg38)[match(DF$C, names(seqlengths(BSgenome.Hsapiens.UCSC.hg38)))]
  DF <- DF[
    (as.numeric(as.character(DF$D)) - (halfw - 1)) > 0 &
    (as.numeric(as.character(DF$D)) + halfw) < chrends &
    (!is.na(chrends)),
  ]

  if (filterCoding) {
    input <- GRanges(
      seqnames = Rle(DF$C),
      ranges   = IRanges(DF$D, end = DF$D),
      strand   = Rle(strand("*"))
    )
    loc <- locateVariants(input,
                          TxDb.Hsapiens.UCSC.hg38.knownGene,
                          CodingVariants())
    DF <- DF[
      is.na(
        match(
          paste(DF$C, DF$D),
          unique(paste(seqnames(loc), start(loc)))
        )
      ),
    ]
  }

  write.csv(DF, file = out_file, row.names = FALSE)
}

data <- fread(in_file, skip = "#")
process.simple.offsets.python(data, window = 2788, out_file = out_file)
