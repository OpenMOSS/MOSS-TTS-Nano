import org.gradle.api.tasks.testing.Test

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.openmoss.ttsnano.onnxruntime"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.openmoss.ttsnano.onnxruntime"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")

    testImplementation("junit:junit:4.13.2")
}

tasks.withType<Test>().configureEach {
    inputs.property("MOSS_TOKENIZER_MODEL", System.getenv("MOSS_TOKENIZER_MODEL").orEmpty())
}
