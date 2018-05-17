import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

buildscript {
    var kotlin_version: String by extra
    kotlin_version = "1.2.41"

    repositories {
        mavenCentral()
    }
    dependencies {
        classpath(kotlinModule("gradle-plugin", kotlin_version))
    }
}

plugins {
    java
    application
}

group = "com.github.holgerbrandl.dl"
version = "1.0-SNAPSHOT"

apply {
    plugin("kotlin")
}

// https://stackoverflow.com/a/50045271/590437
application {
    //    mainClassName = "com.github.holgerbrandl.Tester"
    //    mainClassName = "VggTransferKt"
    mainClassName = "MnistClassifier"
}

val kotlin_version: String by extra

repositories {
    mavenCentral()
    mavenLocal()
    jcenter()
    maven("http://dl.bintray.com/kyonifer/maven")
}

//val nd4jVersion = "1.0.0-alpha"
val nd4jVersion = "1.0.0-SNAPSHOT"


dependencies {
    compile(kotlinModule("stdlib-jdk8", kotlin_version))

//    compile("org.nd4j","nd4j-native-platform","1.0.0-alpha")
    compile("org.nd4j", "nd4j-native-platform", "1.0.0-SNAPSHOT")
    //    compile(     "org.nd4j","nd4j-native",nd4jVersion)
    compile("org.nd4j", "nd4j-native", nd4jVersion, classifier = "macosx-x86_64")

    //    compile("org.nd4j","nd4j-cuda-8.0-platform","1.0.0-alpha")

    compile("org.nd4j","nd4s_2.11","0.7.2")
    compile("org.deeplearning4j", "deeplearning4j-core", nd4jVersion)
    compile("org.deeplearning4j", "deeplearning4j-zoo", nd4jVersion)
    compile("org.deeplearning4j", "deeplearning4j-nn", nd4jVersion)

    // just needed for iris on claspath
    //    compile("org.deeplearning4j","dl4j-test-resources", "1.0.0-SNAPSHOT")

    // http://saltnlight5.blogspot.de/2013/08/how-to-configumacosx-x86_64re-slf4j-with-different.html
    // compile("org.slf4j:slf4j-simple:1.7.25")
    compile("org.slf4j", "slf4j-jdk14", "1.7.5")
    compile("org.apache.httpcomponents", "httpclient", "4.3.5")

    //    compile("de.mpicbg.scicomp", "krangl", "0.10-SNAPSHOT")
    compile("de.mpicbg.scicomp", "krangl", "0.9.1")
    //    compile("com.github.holgerbrandl", "kravis", "0.1-SNAPSHOT")
    compile("com.github.holgerbrandl", "kravis", "0.2")

    compile("koma", "core", "0.11")


    testCompile("junit", "junit", "4.12")
}

configure<JavaPluginConvention> {
    sourceCompatibility = JavaVersion.VERSION_1_8
}
tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
}


// print the classpath
//println(configurations.runtime.resolve())
