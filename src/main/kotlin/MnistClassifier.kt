import DataUtilities.downloadFile
import DataUtilities.extractTarGz
import org.apache.commons.compress.archivers.tar.TarArchiveEntry
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import java.util.HashMap
import java.util.Random

import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.schedule.MapSchedule
import org.nd4j.linalg.schedule.ScheduleType
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import java.io.*

/**
 * Handwritten digits image classification on MNIST dataset (99% accuracy).
 * This example will download 15 Mb of data on the first run.
 * Supervised learning best modeled by CNN.
 *
 * @author hanlon
 * @author agibsonccc
 * @author fvaleri
 */
object MnistClassifier {

    private val log = LoggerFactory.getLogger(MnistClassifier::class.java)
    private val basePath = System.getProperty("java.io.tmpdir") + "/mnist"
    private val dataUrl = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz"

    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        val height = 28
        val width = 28
        val channels = 1 // single channel for grayscale images
        val outputNum = 10 // 10 digits classification
        val batchSize = 54
        val nEpochs = 1
        val iterations = 1

        val seed = 1234
        val randNumGen = Random(seed.toLong())

        log.info("Data load and vectorization...")
        val localFilePath = "$basePath/mnist_png.tar.gz"
        if (downloadFile(dataUrl, localFilePath))
            log.debug("Data downloaded from {}", dataUrl)
        if (!File("$basePath/mnist_png").exists())
            extractTarGz(localFilePath, basePath)

        // vectorization of train data
        val trainData = File("$basePath/mnist_png/training")
        val trainSplit = FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
        val labelMaker = ParentPathLabelGenerator() // parent path as the image label
        val trainRR = ImageRecordReader(height, width, channels, labelMaker)
        trainRR.initialize(trainSplit)
        val trainIter = RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum)

        // pixel values from 0-255 to 0-1 (min-max scaling)
        val scaler = ImagePreProcessingScaler(0.0, 1.0)
        scaler.fit(trainIter)
        trainIter.preProcessor = scaler

        // vectorization of test data
        val testData = File("$basePath/mnist_png/testing")
        val testSplit = FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
        val testRR = ImageRecordReader(height, width, channels, labelMaker)
        testRR.initialize(testSplit)
        val testIter = RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum)
        testIter.preProcessor = scaler // same normalization for better results

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
            .setInputType(InputType.convolutionalFlat(28, 28, 1)) // InputType.convolutional for normal image
            .backprop(true).pretrain(false).build()

        val net = MultiLayerNetwork(conf)
        net.init()
        net.setListeners(ScoreIterationListener(10))
        log.debug("Total num of params: {}", net.numParams())

        // evaluation while training (the score should go down)
        for (i in 0 until nEpochs) {
            net.fit(trainIter)
            log.info("Completed epoch {}", i)
            val eval = net.evaluate(testIter)
            log.info(eval.stats())
            trainIter.reset()
            testIter.reset()
        }

        ModelSerializer.writeModel(net, File("$basePath/minist-model.zip"), true)
    }
}
