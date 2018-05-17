package so

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.transform.TransformProcess
import org.datavec.api.transform.schema.Schema
import java.io.File
import java.net.URL
import java.util.*


/**
 * https://stackoverflow.com/questions/50374063/how-do-i-remove-dummy-variable-trap-with-onehotencoding
 *
 * @author Holger Brandl
 */
fun main(args: Array<String>) {
    val schema = Schema.Builder()
        .addColumnsString("RowNumber")
        .addColumnInteger("CustomerId")
        .addColumnString("Surname")
        .addColumnInteger("CreditScore")
        .addColumnCategorical("Geography", Arrays.asList("France", "Spain", "Germany"))
        .addColumnCategorical("Gender", Arrays.asList("Male", "Female"))
        .addColumnsInteger("Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited").build()

    val transformProcess = TransformProcess.Builder(schema)
        .removeColumns("RowNumber", "Surname", "CustomerId")
        .categoricalToInteger("Gender")
        .categoricalToOneHot("Geography")
        .removeColumns("Geography[France]")
        .build()

    //    transformProcess.finalSchema
    val reader = CSVRecordReader(1, ',')

    val dataFile = fileOfUrl(URL("https://raw.githubusercontent.com/2blam/ML/master/deep_learning/Churn_Modelling.csv"))
    reader.initialize(FileSplit(dataFile))

    //    reader.initialize(FileSplit(ClassPathResource("Churn_Modelling.csv").getFile()))


    val transformProcessRecordReader = TransformProcessRecordReader(reader, transformProcess)
    println("args = " + transformProcessRecordReader.next() + "")
}

private fun fileOfUrl(url: URL): File {
    val localFile = File(File(url.file).name)
    if (!localFile.exists()) url.readBytes().run { localFile.also { it.writeBytes(this) } }
    return localFile
}