// ~/projects/kotlin/misc/sparklin/bin/kshell.sh
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import java.io.File

//println(MNIST_DATA_ROOT)

//val tt = createTrainDataIt()
//tt.next()

val DATA_ROOT = File("${System.getProperty("user.home")}/.kaggle/competitions/digit-recognizer/")


val dataFile = File(DATA_ROOT, "train.csv")

val recordReader = CSVRecordReader(1, ',').apply {
    val fileSplit = FileSplit(dataFile)
    initialize(fileSplit)
}

recordReader.next()

val iter = RecordReaderDataSetIterator.Builder(recordReader, 64).classification(0, 10).build()


val nextOne = iter.next()
println(nextOne)

val foo = createTrainDataIt().next()

reader2().next()

//val cl = ClassLoader.getSystemClassLoader();
//val urls = (INDArray::class.java.classLoader as URLClassLoader).getURLs();
//println(urls.joinToString(","))
