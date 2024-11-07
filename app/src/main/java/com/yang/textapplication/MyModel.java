package com.yang.textapplication;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.Environment;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;


import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public class MyModel {

    Context mContext;
    private Interpreter tflite;
    private GpuDelegate gpuDelegate;  // 添加GPU Delegate
    private static final String MODEL_PATH = "model.tflite";  // 修改为实际模型路径
    private static final int MAX_LENGTH = 128;

    // 初始化模型并启用GPU
    public void init(Context context) {
        mContext = context;
        try {
            // 加载模型文件
            MappedByteBuffer modelBuffer = loadModelFile(context, MODEL_PATH);

            // 创建 GPU Delegate
            GpuDelegate.Options options = new GpuDelegate.Options();
            gpuDelegate = new GpuDelegate(options);

            // 使用 GPU Delegate 创建 Interpreter
            Interpreter.Options interpreterOptions = new Interpreter.Options().addDelegate(gpuDelegate);
            tflite = new Interpreter(modelBuffer, interpreterOptions);

            Log.d("MyModel", "模型加载成功，GPU推理已启用！");
        } catch (IOException e) {
            Log.e("MyModel", "模型加载失败: " + e.getMessage());
        }
    }

    // 使用 MappedByteBuffer 从 assets 目录加载模型文件
    private MappedByteBuffer loadModelFile(Context context, String modelName) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        MappedByteBuffer mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

        inputStream.close();
        return mappedByteBuffer;
    }


    public float runModel(long[][] input_ids, long[][] attention_mask, long[][] token_type_ids) {
        if (tflite == null) {
            Log.e("MyModel", "TensorFlow Lite 模型尚未加载！");
            return 0;
        }

        int inputCount = tflite.getInputTensorCount();
        for (int i = 0; i < inputCount; i++) {
            Tensor inputTensor = tflite.getInputTensor(i);

            // 获取详细的输入信息
            String tensorName = inputTensor.name();  // 张量名称
            String tensorShape = Arrays.toString(inputTensor.shape());  // 张量形状
            String tensorDataType = inputTensor.dataType().toString();  // 数据类型
            String tensorQuantization = inputTensor.quantizationParams().toString();  // 量化参数
            int numElements = inputTensor.numElements();  // 张量的元素数量
            int tensorIndex = inputTensor.index();  // 张量的索引

            // 打印日志
            Log.d("Model Input", "Input " + i + " (Index " + tensorIndex + "): " +
                    "Name: " + tensorName + ", " +
                    "Shape: " + tensorShape + ", " +
                    "Data Type: " + tensorDataType + ", " +
                    "Quantization: " + tensorQuantization + ", " +
                    "Num Elements: " + numElements);
        }

        int outputCount = tflite.getOutputTensorCount();
        for (int i = 0; i < outputCount; i++) {
            Tensor outputTensor = tflite.getOutputTensor(i);

            // 获取详细的输出信息
            String tensorName = outputTensor.name();  // 张量名称
            String tensorShape = Arrays.toString(outputTensor.shape());  // 张量形状
            String tensorDataType = outputTensor.dataType().toString();  // 数据类型
            String tensorQuantization = outputTensor.quantizationParams().toString();  // 量化参数
            int numElements = outputTensor.numElements();  // 张量的元素数量
            int tensorIndex = outputTensor.index();  // 张量的索引

            // 打印日志
            Log.d("Model Output", "Output " + i + " (Index " + tensorIndex + "): " +
                    "Name: " + tensorName + ", " +
                    "Shape: " + tensorShape + ", " +
                    "Data Type: " + tensorDataType + ", " +
                    "Quantization: " + tensorQuantization + ", " +
                    "Num Elements: " + numElements);
        }


        Object[] inputs = {attention_mask, input_ids, token_type_ids};
        float[][] output0Float = new float[2][1024];
        float[][][] output1Float = new float[2][128][1024];


        /*Map<String, Object> mapInputs = new HashMap<>();
        mapInputs.put("attention_mask", attention_mask);
        mapInputs.put("input_ids", input_ids);
        mapInputs.put("token_type_ids", token_type_ids);
        Map<String, Object> mapOutput = new HashMap<>();
        mapOutput.put("1525", output0Float);  // 使用输出0的索引
        mapOutput.put("last_hidden_state", output1Float);  // 使用输出1的索引
        tflite.runSignature(mapInputs, mapOutput);*/



        Map<Integer, Object> map = new HashMap<>();
        map.put(1, output0Float);  // 使用输出0的索引
        map.put(0, output1Float);  // 使用输出1的索引


        tflite.runForMultipleInputsOutputs(inputs, map);

        return computeCosineSimilarity(flatten(output1Float[0]), flatten(output1Float[1]));
    }


    public float computeCosineSimilarity(float[] vec1, float[] vec2) {

        saveVectorsToFile(mContext, vec1, vec2);
        // 计算向量的范数（L2范数），使用double类型提高精度
        double normVec1 = norm(vec1);
        double normVec2 = norm(vec2);

        // 防止除以零的情况
        if (normVec1 == 0 || normVec2 == 0) {
            return 0.0f;  // 如果其中一个向量的范数为零，余弦相似度为0
        } else {
            // 计算点积，使用double类型
            double dotProduct = dotProduct(vec1, vec2);
            // 返回余弦相似度，转换为float类型
            return (float) (dotProduct / (normVec1 * normVec2));
        }
    }

    // 计算向量的范数（L2范数），使用double类型
    public static double norm(float[] vec) {
        double sum = 0.0;
        for (float val : vec) {
            sum += val * val;
        }
        return Math.sqrt(sum);
    }

    // 计算向量的点积，使用double类型
    public static double dotProduct(float[] vec1, float[] vec2) {
        double sum = 0.0;
        for (int i = 0; i < vec1.length; i++) {
            sum += vec1[i] * vec2[i];
        }
        return sum;
    }

    // 释放资源时关闭 GPU Delegate
    public void close() {
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
            Log.d("MyModel", "GPU Delegate 已释放！");
        }

        if (tflite != null) {
            tflite.close();
            tflite = null;
            Log.d("MyModel", "TensorFlow Lite 模型已释放！");
        }
    }


    private long[][] convertToLongArray(long[][] input) {
        int targetLength = 128;  // 目标行长度为128
        long[][] longArray = new long[input.length][targetLength];

        for (int i = 0; i < input.length; i++) {
            // 获取当前行的长度
            int currentLength = input[i].length;

            // 复制当前行的数据到新的数组
            for (int j = 0; j < currentLength; j++) {
                longArray[i][j] = input[i][j];
            }

            // 如果当前行的长度不足128，则填充其余部分
            for (int j = currentLength; j < targetLength; j++) {
                longArray[i][j] = 0;  // 填充值可以根据需要修改
            }
        }

        return longArray;
    }

    public float[] flatten(float[][] array) {
        // 获取二维数组的总元素数
        int totalLength = 0;
        for (float[] row : array) {
            totalLength += row.length;
        }

        // 创建一维数组
        float[] flattenedArray = new float[totalLength];

        // 填充一维数组
        int index = 0;
        for (float[] row : array) {
            for (float value : row) {
                flattenedArray[index++] = value;
            }
        }

        return flattenedArray;
    }


    public void saveVectorsToFile(Context context, float[] vec1, float[] vec2) {
        StringBuilder sb = new StringBuilder();

        // 将 vec1 和 vec2 转换为字符串格式
        sb.append("vec1: ").append(arrayToString(vec1)).append("\n");
        sb.append("vec2: ").append(arrayToString(vec2)).append("\n");

        // 创建文件
        File file = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "vectors.txt");

        try (FileOutputStream fos = new FileOutputStream(file)) {
            fos.write(sb.toString().getBytes());
            Log.d("MyModel", "Vectors saved to " + file.getAbsolutePath());
        } catch (IOException e) {
            Log.e("MyModel", "Error saving vectors to file: " + e.getMessage());
        }
    }

    // 辅助方法：将 float 数组转化为字符串
    private String arrayToString(float[] array) {
        StringBuilder sb = new StringBuilder();
        for (float value : array) {
            sb.append(value).append(",");
        }
        return sb.toString();
    }
}
