package com.yang.textapplication;

import android.content.Context;
import android.content.res.AssetManager;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

public class VocabLoader {
    public static Map<String, Integer> loadVocab(Context context, String vocabFilePath) throws IOException {
        Map<String, Integer> vocab = new HashMap<>();
        // 使用 AssetManager 读取文件
        AssetManager assetManager = context.getAssets();
        InputStream inputStream = assetManager.open(vocabFilePath);

        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
        String line;
        int index = 0;
        while ((line = reader.readLine()) != null) {
            vocab.put(line, index++);
        }
        reader.close();
        return vocab;
    }
}
