// ~/projects/kotlin/misc/sparklin/bin/kshell.sh
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import java.io.File

//println(MNIST_DATA_ROOT)

fun buildIterator(): RecordReaderDataSetIterator {
    val dataFile = File("train.split_${DataSplit.TRAIN.name.toLowerCase()}.csv")

    val recordReader = CSVRecordReader(1, ',').apply {
        val fileSplit = FileSplit(dataFile)

        //        val splitInput = fileSplit.sample(null, 0.7, 0.2, 0.1)
        initialize(fileSplit)
    }

    recordReader.next()

    val iter = RecordReaderDataSetIterator.Builder(recordReader, 128)
        .classification(0, 10)
        //        .preProcessor(NormalizerStandardize()).build()

        // @AlexDBlack: normally a .setInputType(InputType.convolutionalFlat(...)) is all you need for that...
        //        .preProcessor(object : DataSetPreProcessor {
        //            override fun preProcess(ds: DataSet) {
        //                println("shape is ${ds.features.shape().joinToString(",")}")
        //                //                val zScaled = NormalizerStandardize().transform(ds)
        //                val curBatchSize = ds.features.shape()[0]
        //                ds.features = ds.features.reshape(ds.features.shape()[0], 1, 28, 28)
        //                ds.features = ds.features.run{reshape(shape()[0], 1, 28, 28)}
        ////                NormalizerMinMaxScaler().transform(ds)
        //            }
        //        })
        //    .preProcessor {
        //        // https://deeplearning4j.org/core-concepts#normalizing-data
        //        NormalizerStandardize()
        //            .apply { fit(it) }
        //            .transform(it)
        //    }
        .build()

    return iter
}

val iter = buildIterator()

println(iter.next())

//tt.if

