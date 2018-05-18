import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.BaseTrainingListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.schedule.MapSchedule
import org.nd4j.linalg.schedule.ScheduleType
import org.slf4j.LoggerFactory
import java.io.File
import java.nio.file.Files
import java.util.*
import kotlin.math.roundToInt
import kotlin.system.exitProcess

/*
 * Adopted from https://github.com/holgerbrandl/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/mnist/MnistClassifier.java
 *
 * Handwritten digits image classification on MNIST dataset (99% accuracy).
 * This example will download 15 Mb of data on the first run.
 * Supervised learning best modeled by CNN.
 *
 * @author hanlon
 * @author agibsonccc
 * @author fvaleri
 * @author holgerbrandl
 */

private val log = LoggerFactory.getLogger(MnistClassifier::class.java)

val MNIST_DATA_ROOT = File("${System.getProperty("user.home")}/.kaggle/competitions/digit-recognizer/")


object MnistClassifier {

    //    private val dataUrl = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz"


    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        val height = 28
        val width = 28
        val channels = 1 // single channel for grayscale images
        val outputNum = 10 // 10 digits classification
        val nEpochs = 1
        val batchSize = 128


        File(System.getProperty("user.dir"), "test.txt").writeText("hello from in here")

        val seed = 1234
        //        val randNumGen = Random(seed.toLong())

        log.info("Data load and vectorization...")

        if (!MNIST_DATA_ROOT.isDirectory) {
            println("Kaggle data is not yet downloaded. See REAMDE.md ")
            exitProcess(-1)
        }

        val trainIter = createTrainDataIt(batchSize = batchSize, dataSet = DataSplit.TRAIN)
        //        val feature = trainIter.next().features.slice(0)

        //    // plot one of them
        //    val next = iter.next()
        //    val image = next.features.reshape(10, 28, 28).slice(7).toDoubleMatrix()
        //    val heatmapData = image.withIndex()
        //        .map { (index, data) -> DoubleCol(index.toString(), data.toList()) }.bindCols()
        //        .addRowNumber("y")
        //        .gather("x", "pixel_value", { except("y") }, convert = true)
        //
        //    heatmapData.heatmap("x", "y", "pixel_value")


        // https://www.researchgate.net/figure/Slices-of-a-3rd-order-tensor_fig2_251235488
        //        val koma = create(feature.reshape(10, 28, 28).slice(0).toDoubleMatrix())
        //        plot(koma, 'b', "First Run")


        //        feature.reshape(10, 28, 28)

        //        val next = trainIt.next()
        ////        next.
        //
        //        println("${next.features}")
        //        // vectorization of train data
        //        val trainData = File("$basePath/mnist_png/training")
        //        val trainSplit = FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
        //        val labelMaker = ParentPathLabelGenerator() // parent path as the image label
        //        val trainRR = ImageRecordReader(height, width, channels, labelMaker)
        //        trainRR.initialize(trainSplit)
        //        val trainIter = RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum)
        //
        //        // pixel values from 0-255 to 0-1 (min-max scaling)
        //        val scaler = ImagePreProcessingScaler(0.0, 1.0)
        //        scaler.fit(trainIter)
        //        trainIter.preProcessor = scaler
        //
        //        // vectorization of test data
        //        val testData = File("$basePath/mnist_png/testing")
        //        val testSplit = FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
        //        val testRR = ImageRecordReader(height, width, channels, labelMaker)
        //        testRR.initialize(testSplit)
        //        val testIter = RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum)
        //        testIter.preProcessor = scaler // same normalization for better results

        // note: model configuration is just a copy from the dl4j examples

        log.info("Network configuration and training...")
        val lrSchedule = HashMap<Int, Double>()
        lrSchedule[0] = 0.06 // iteration #, learning rate
        lrSchedule[200] = 0.05
        lrSchedule[600] = 0.028
        lrSchedule[800] = 0.0060
        lrSchedule[1000] = 0.001

        val conf = NeuralNetConfiguration.Builder()
            .seed(seed.toLong())
            .l2(0.0005)
            .updater(Nesterovs(MapSchedule(ScheduleType.ITERATION, lrSchedule)))
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(0, ConvolutionLayer.Builder(5, 5)
                .nIn(channels)
                .stride(1, 1)
                .nOut(20)
                .activation(Activation.IDENTITY)
                .build())
            .layer(1, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(2, ConvolutionLayer.Builder(5, 5)
                .stride(1, 1) // nIn need not specified in later layers
                .nOut(50)
                .activation(Activation.IDENTITY)
                .build())
            .layer(3, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(4, DenseLayer.Builder().activation(Activation.RELU)
                .nOut(500).build())
            .layer(5, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .build())
            // normally a .setInputType(InputType.convolutionalFlat(...)) is all you need for that...  it adds a preprocessor to do the reshaping
            .setInputType(InputType.convolutionalFlat(height, width, channels)) // InputType.convolutional for normal image
            //            .setInputType(InputType.convolutional(height, width, channels)) // InputType.convolutional for normal image
            .backprop(true).pretrain(false)
            .build()

        val net = MultiLayerNetwork(conf).apply {
            init()

            addListeners(ScoreIterationListener(10))

            addListeners(object : BaseTrainingListener() {
                override fun onEpochEnd(model: Model?) {
                    log.info("Completed epoch ${(model as MultiLayerNetwork).epochCount}")
                }
            })

            //            addListeners(object : BaseTrainingListener() {
            //                override fun onEpochEnd(model: Model?) {
            //                    val model = model as MultiLayerNetwork
            //                    val eval = model.evaluate(testIter)
            //                    log.info(eval.stats())
            //                    testIter.reset()
            //                }
            //            })
        }

        log.debug("Total num of params: {}", net.numParams())

        //        net.params().toDoubleMatrix().

        // evaluation while training (the score should go down)
        net.fit(trainIter, nEpochs)

        ModelSerializer.writeModel(net, File("kaggle-mnist-model.zip"), true)


        //        log.info("Evaluate model....")
        //        val eval = Evaluation(outputNum) //create an evaluation object with 10 possible classes
        //        while (mnistTest.hasNext()) {
        //            val next = mnistTest.next()
        //            val output = model.output(next.features) //get the networks prediction
        //            eval.eval(next.getLabels(), output) //check the prediction against the true class
        //        }

    }
}


object ModelEval {

    @JvmStatic
    fun main(args: Array<String>) {
        //        ModelSerializer.writeModel(net, File("kaggle-mnist-model.zip"), true)

        val model = MultiLayerNetwork.load(File("kaggle-mnist-model.zip"), false)

        val mnistTest = createTrainDataIt(dataSet = DataSplit.TEST)


        log.info("Evaluate model....")
        val eval = Evaluation(10) //create an evaluation object with 10 possible classes
        while (mnistTest.hasNext()) {
            mnistTest.reset()
            val next = mnistTest.next()
            val output = model.output(next.features) //get the networks prediction
            eval.eval(next.getLabels(), output) //check the prediction against the true class
        }


        println(eval.confusionMatrix.toHTML())

        //http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        // weighted average for multi-class problems
        println("${eval.f1()}")

        println(eval.f1())
    }
}


object PrepareSubmission {


    fun createTestIter(dataFile: File): DataSetIterator {
        require(dataFile.exists()) { "data file $dataFile does not exist" }

        val recordReader = CSVRecordReader(1, ',').apply {
            initialize(FileSplit(dataFile))
        }


        val iter = RecordReaderDataSetIterator.Builder(recordReader, 128)
            //            .classification(-1, 10)
            //        .classification(0, 10)
            .preProcessor {
                // https://deeplearning4j.org/core-concepts#normalizing-data
                NormalizerStandardize()
                    .apply { fit(it) }
                    .transform(it)
            }
            .build()

        return iter
    }


    @JvmStatic
    fun main(args: Array<String>) {
        val model = MultiLayerNetwork.load(File("kaggle-mnist-model.zip"), false)

        val submissionTestFile = File(MNIST_DATA_ROOT, "test.csv")
        val testIter = createTestIter(submissionTestFile)

        log.info("Processing submission test data....")

        val results = testIter.asSequence().map { ds ->
            ds.labelNames = (1..10).map { it.toString() }

            model.predict(ds)
        }.flatten()

        // expected format
        //        ImageId,Label
        //        1,0
        //        2,0
        //        3,0

        val submissionFile = File("kaggle_mnist_submission.txt")

        submissionFile.printWriter().use { pw ->
            // the submission format should be business_id to labels
            pw.write("ImageId\tLabel\n")

            results.withIndex().forEach {
                pw.write("${it.index},${it.value}\n")
            }
        }

        val totalRecords = Files.lines(submissionTestFile.toPath()).count();
        val totalSubmission = Files.lines(submissionFile.toPath()).count();

        print("total: ${totalRecords} vs submissions ${totalSubmission}")
    }
}

enum class DataSplit { TRAIN, CV, TEST }

fun createTrainDataIt(
    batchSize: Int = 128,
    //    maxExamples: Int = Int.MAX_VALUE,
    dataSet: DataSplit = DataSplit.TRAIN

): DataSetIterator {


    //    var iter2: MultiDataSetIterator = RecordReaderMultiDataSetIterator.Builder(10).addReader("mnist_reader", recordReader)
    //        .addInput("mnist_reader", 1, 784)
    //        .addOutputOneHot("mnist_reader", 0, 10)
    //        .build()
    //
    //    iter.next().features

    // line format
    // label,pixel0,pixel1,pixel2,pixel3,pixel4,pixel5, ... pixel783

    //    val DATA_ROOT = File("${System.getProperty("user.home")}/.kaggle/competitions/digit-recognizer/")

    //    val dataFile = File("mnist_subset.csv")
    //            .also {
    //                it.printWriter().let { pw ->
    //                    File(DATA_ROOT, "train.csv").readLines().take(500).forEach { pw.println(it) }
    //                    pw.close()
    //                }
    //            }

    //    DataFrame.readCSV(dataFile).count("label").print()

    //// actual kaggle set without any cross-val subset
    //    val dataFile = File(MNIST_DATA_ROOT, "train.csv")

    val dataFile = File("train.split_${dataSet.name.toLowerCase()}.csv")
    require(dataFile.exists()) { "data file ${dataFile} does not exist" }


    val recordReader = CSVRecordReader(1, ',').apply {
        val fileSplit = FileSplit(dataFile)

        //        val splitInput = fileSplit.sample(null, 0.7, 0.2, 0.1)
        initialize(fileSplit)
    }


    val iter = RecordReaderDataSetIterator.Builder(recordReader, batchSize)
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
        .preProcessor {
            // https://deeplearning4j.org/core-concepts#normalizing-data
            NormalizerStandardize()
                .apply { fit(it) }
                .transform(it)
        }
        .build()

    //    TransformProcessRecordReader(recordReader, object : TransformProcess() {})


    return iter
    //    return ExistingD    ataSetIterator(iter)
}


// AlexDBlack: for something like CSV, you usually split it ahead of time - i.e., create new CSVs, one for training, one for test
object TrainDataSplit {

    @JvmStatic
    fun main(args: Array<String>) {

        fun writeTrainSplit(dataFile: File, it: Sequence<String>, subset: String) {
            File(dataFile.nameWithoutExtension + ".split_$subset.csv").writeText(it.joinToString("\n"))
        }


        //         split train.csv manually into train, validation and test set
        val dataFile = File(MNIST_DATA_ROOT, "train.csv")

        //    val totalRecords = Files.lines(dataFile.toPath()).count();


        dataFile.readLines()
            //            .shuffled()
            .run {
                val iter = iterator()

                iter.asSequence().take((size * 0.7).roundToInt()).let { writeTrainSplit(dataFile, it, "train") }
                iter.asSequence().take((size * 0.15).roundToInt()).let { writeTrainSplit(dataFile, it, "cv") }
                iter.asSequence().let { writeTrainSplit(dataFile, it, "test") }
            }
    }
}