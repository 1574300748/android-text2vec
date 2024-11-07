package com.yang.textapplication;

import java.util.List;
import java.util.Map;

public class Tokenizer {

    // 修改了返回类型：返回 int[][]，即一个包含三个一维数组的二维数组
    public static long[][] convertToIds(String[] tokens, Map<String, Integer> vocab, int maxLength) {
        long[] inputIds = new long[maxLength];
        long[] attentionMask = new long[maxLength];
        long[] tokenTypeIds = new long[maxLength];  // tokenTypeIds 全部填充为 0

        // 设置特殊tokens
        inputIds[0] = 101;  // 起始token ID [CLS]
        attentionMask[0] = 1;
        tokenTypeIds[0] = 0;  // 设置 tokenTypeIds 的初始值为 0

        // 填充token ids
        int index = 1;
        for (String token : tokens) {
            if (vocab.containsKey(token)) {
                inputIds[index] = vocab.get(token);
                attentionMask[index] = 1;
                tokenTypeIds[index] = 0;  // 始终将 tokenTypeIds 填充为 0
            }
            index++;
            if (index >= maxLength - 1) break; // 防止溢出
        }

        // 在有效部分后面添加 [SEP] (102)
        if (index < maxLength) {
            inputIds[index] = 102;  // 结束token ID [SEP]
            attentionMask[index] = 1;
            tokenTypeIds[index] = 0;  // 确保 [SEP] 对应的 tokenTypeIds 也为 0
            index++;
        }

        // 填充剩余部分为0
        for (int i = index; i < maxLength; i++) {
            inputIds[i] = 0; // padding token ID
            attentionMask[i] = 0;
            tokenTypeIds[i] = 0;  // tokenTypeIds 填充为 0
        }

        // 返回二维数组
        return new long[][]{inputIds, attentionMask, tokenTypeIds};
    }
}
