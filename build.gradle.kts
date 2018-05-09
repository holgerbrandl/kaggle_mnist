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
}

val nd4jVersion = "1.0.0-alpha"


dependencies {
    compile(kotlinModule("stdlib-jdk8", kotlin_version))

//    compile("org.nd4j","nd4j-native-platform","1.0.0-alpha")
    compile("org.nd4j","nd4j-cuda-8.0-platform","1.0.0-alpha")
    compile("org.nd4j","nd4s_2.11","0.7.2")
    compile("org.deeplearning4j", "deeplearning4j-core", nd4jVersion)
    compile("org.deeplearning4j", "deeplearning4j-zoo", nd4jVersion)
    compile("org.deeplearning4j", "deeplearning4j-nn", nd4jVersion)

    // http://saltnlight5.blogspot.de/2013/08/how-to-configure-slf4j-with-different.html
    // compile("org.slf4j:slf4j-simple:1.7.25")
    compile("org.slf4j", "slf4j-jdk14", "1.7.5")
    compile("org.apache.httpcomponents", "httpclient", "4.3.5")

    testCompile("junit", "junit", "4.12")
}

configure<JavaPluginConvention> {
    sourceCompatibility = JavaVersion.VERSION_1_8
}
tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
}