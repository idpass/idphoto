# Java SDK

Package namespace: `org.idpass.idphoto`

Primary wrapper: `JavaIdPhoto` (`bindings/jni/kotlin/JavaIdPhoto.kt`)

## Build

```bash
cargo build -p idphoto-jni
```

## API

Use the Java static facade and Java-specific option builders:

- `JavaIdPhoto.compress(data)`
- `JavaIdPhoto.compress(data, options)`
- `JavaIdPhoto.compressToFit(data, maxBytes)`
- `JavaIdPhoto.compressToFit(data, maxBytes, options)`

```java
import org.idpass.idphoto.*;

byte[] raw = Files.readAllBytes(Path.of("photo.jpg"));

JavaCompressOptions options = new JavaCompressOptions.Builder()
    .setPreset(Preset.QR_CODE)
    .setFormat(OutputFormat.JPEG)
    .build();

JavaCompressedPhoto result = JavaIdPhoto.compress(raw, options);
System.out.println(JavaIdPhoto.summary(result));

JavaCompressToFitOptions fitOptions = new JavaCompressToFitOptions.Builder()
    .setPreset(Preset.QR_CODE)
    .build();

JavaFitResult fit = JavaIdPhoto.compressToFit(raw, 2048L, fitOptions);
System.out.println(JavaIdPhoto.summary(fit));
```

## Java Wrapper Types

`JavaIdPhoto` returns Java-friendly model types:

- `JavaCompressedPhoto`
- `JavaFitResult`
- `JavaCompressOptions` + `JavaCompressOptions.Builder`
- `JavaCompressToFitOptions` + `JavaCompressToFitOptions.Builder`

These wrappers hide unsigned Kotlin numeric types from Java call sites.

## Utility Methods

- `JavaIdPhoto.mimeType(JavaCompressedPhoto)`
- `JavaIdPhoto.fileExtension(JavaCompressedPhoto)`
- `JavaIdPhoto.summary(JavaCompressedPhoto)`
- `JavaIdPhoto.summary(JavaFitResult)`

## Advanced

For lower-level generated bindings and type mapping details, see
[JVM Low-Level API (UniFFI/JNI)](jni.md).
