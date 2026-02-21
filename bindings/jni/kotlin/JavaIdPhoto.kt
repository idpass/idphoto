package org.idpass.idphoto

/**
 * Java-friendly options for [JavaIdPhoto.compress].
 *
 * Uses Java primitive types and boxed nullable values instead of unsigned
 * Kotlin numeric types from the generated bindings.
 */
class JavaCompressOptions private constructor(
    private val preset: Preset?,
    private val maxDimension: Int?,
    private val quality: Float?,
    private val grayscale: Boolean?,
    private val cropMode: CropMode?,
    private val format: OutputFormat?,
    private val faceMargin: Float?,
) {
    class Builder {
        private var preset: Preset? = null
        private var maxDimension: Int? = null
        private var quality: Float? = null
        private var grayscale: Boolean? = null
        private var cropMode: CropMode? = null
        private var format: OutputFormat? = null
        private var faceMargin: Float? = null

        fun setPreset(value: Preset?) = apply { preset = value }
        fun setMaxDimension(value: Int?) = apply { maxDimension = value }
        fun setQuality(value: Float?) = apply { quality = value }
        fun setGrayscale(value: Boolean?) = apply { grayscale = value }
        fun setCropMode(value: CropMode?) = apply { cropMode = value }
        fun setFormat(value: OutputFormat?) = apply { format = value }
        fun setFaceMargin(value: Float?) = apply { faceMargin = value }

        fun build(): JavaCompressOptions = JavaCompressOptions(
            preset = preset,
            maxDimension = maxDimension,
            quality = quality,
            grayscale = grayscale,
            cropMode = cropMode,
            format = format,
            faceMargin = faceMargin,
        )
    }

    internal fun toNative(): CompressOptions = CompressOptions(
        preset = preset,
        maxDimension = maxDimension?.toUInt(),
        quality = quality,
        grayscale = grayscale,
        cropMode = cropMode,
        format = format,
        faceMargin = faceMargin,
    )
}

/**
 * Java-friendly options for [JavaIdPhoto.compressToFit].
 *
 * Mirrors [JavaCompressOptions] except `quality`, which is selected
 * automatically by byte-budget search.
 */
class JavaCompressToFitOptions private constructor(
    private val preset: Preset?,
    private val maxDimension: Int?,
    private val grayscale: Boolean?,
    private val cropMode: CropMode?,
    private val format: OutputFormat?,
    private val faceMargin: Float?,
) {
    class Builder {
        private var preset: Preset? = null
        private var maxDimension: Int? = null
        private var grayscale: Boolean? = null
        private var cropMode: CropMode? = null
        private var format: OutputFormat? = null
        private var faceMargin: Float? = null

        fun setPreset(value: Preset?) = apply { preset = value }
        fun setMaxDimension(value: Int?) = apply { maxDimension = value }
        fun setGrayscale(value: Boolean?) = apply { grayscale = value }
        fun setCropMode(value: CropMode?) = apply { cropMode = value }
        fun setFormat(value: OutputFormat?) = apply { format = value }
        fun setFaceMargin(value: Float?) = apply { faceMargin = value }

        fun build(): JavaCompressToFitOptions = JavaCompressToFitOptions(
            preset = preset,
            maxDimension = maxDimension,
            grayscale = grayscale,
            cropMode = cropMode,
            format = format,
            faceMargin = faceMargin,
        )
    }

    internal fun toNative(): CompressOptions = CompressOptions(
        preset = preset,
        maxDimension = maxDimension?.toUInt(),
        quality = null,
        grayscale = grayscale,
        cropMode = cropMode,
        format = format,
        faceMargin = faceMargin,
    )
}

/**
 * Java-friendly compression result with signed primitive dimensions.
 */
class JavaCompressedPhoto internal constructor(
    data: ByteArray,
    val format: OutputFormat,
    val width: Int,
    val height: Int,
    val originalSize: Long,
    val faceBounds: FaceBounds?,
) {
    val data: ByteArray = data.copyOf()

    override fun toString(): String =
        "JavaCompressedPhoto(${width}x${height}, ${format.name.lowercase()}, ${data.size} bytes, original=${originalSize})"
}

/**
 * Java-friendly byte-budget fit result.
 */
class JavaFitResult internal constructor(
    val photo: JavaCompressedPhoto,
    val qualityUsed: Float,
    val reachedTarget: Boolean,
) {
    override fun toString(): String =
        "JavaFitResult(${photo.width}x${photo.height}, q=${qualityUsed}, reached=${reachedTarget}, ${photo.data.size} bytes)"
}

/**
 * Java-oriented static facade on top of the generated UniFFI API.
 *
 * Kotlin users should prefer [IdPhoto].
 */
object JavaIdPhoto {
    @JvmStatic
    fun compress(data: ByteArray): JavaCompressedPhoto = compress(
        data,
        JavaCompressOptions.Builder().build(),
    )

    @JvmStatic
    fun compress(data: ByteArray, options: JavaCompressOptions?): JavaCompressedPhoto {
        val native = (options ?: JavaCompressOptions.Builder().build()).toNative()
        return org.idpass.idphoto.compress(data, native).toJava()
    }

    @JvmStatic
    fun compressToFit(data: ByteArray, maxBytes: Long): JavaFitResult = compressToFit(
        data,
        maxBytes,
        JavaCompressToFitOptions.Builder().build(),
    )

    @JvmStatic
    fun compressToFit(
        data: ByteArray,
        maxBytes: Long,
        options: JavaCompressToFitOptions?,
    ): JavaFitResult {
        require(maxBytes > 0L) { "maxBytes must be positive" }
        val native = (options ?: JavaCompressToFitOptions.Builder().build()).toNative()
        return org.idpass.idphoto.compressToFit(data, maxBytes.toULong(), native).toJava()
    }

    @JvmStatic
    fun mimeType(photo: JavaCompressedPhoto): String = when (photo.format) {
        OutputFormat.WEBP -> "image/webp"
        OutputFormat.JPEG -> "image/jpeg"
    }

    @JvmStatic
    fun fileExtension(photo: JavaCompressedPhoto): String = when (photo.format) {
        OutputFormat.WEBP -> "webp"
        OutputFormat.JPEG -> "jpg"
    }

    @JvmStatic
    fun summary(photo: JavaCompressedPhoto): String = photo.toString()

    @JvmStatic
    fun summary(fit: JavaFitResult): String = fit.toString()
}

private fun CompressedPhoto.toJava(): JavaCompressedPhoto = JavaCompressedPhoto(
    data = data,
    format = format,
    width = width.toIntExact("width"),
    height = height.toIntExact("height"),
    originalSize = originalSize.toLongExact("originalSize"),
    faceBounds = faceBounds,
)

private fun FitResult.toJava(): JavaFitResult = JavaFitResult(
    photo = photo.toJava(),
    qualityUsed = qualityUsed,
    reachedTarget = reachedTarget,
)

private fun UInt.toIntExact(label: String): Int {
    require(this <= Int.MAX_VALUE.toUInt()) {
        "$label exceeds Int.MAX_VALUE: $this"
    }
    return toInt()
}

private fun ULong.toLongExact(label: String): Long {
    require(this <= Long.MAX_VALUE.toULong()) {
        "$label exceeds Long.MAX_VALUE: $this"
    }
    return toLong()
}
