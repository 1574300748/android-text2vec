package com.yang.textapplication;

import static com.yang.textapplication.Tokenizer.convertToIds;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;


import java.io.IOException;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    private MyModel model;
    private TextView textView;
    private EditText edit_2;
    private EditText edit_1;
    private Button btn;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
        initView();
        model = new MyModel();
        model.init(this);


    }

    public void initView() {
        edit_1 = findViewById(R.id.edit_1);
        edit_2 = findViewById(R.id.edit_2);
        textView = findViewById(R.id.tv_result);
        btn = findViewById(R.id.btn_begin);
        btn.setOnClickListener(v -> {
            submitText(edit_1.getText().toString(), edit_2.getText().toString());
        });
    }

    public void begin(long[][] input1, long[][] input2) {

        float v = model.runModel(new long[][]{input1[0], input2[0]}, new long[][]{input1[1], input2[1]}, new long[][]{input1[2], input2[2]});
        textView.setText("\""+edit_1.getText().toString()+"\"\n\""+edit_2.getText().toString()+"\"\n相似度:"+v+"\n"+(v>0.75?"语义可能相同":"语义可能不同"));
    }

    public void submitText(String text1, String text2) {

        String[] words1 = text1.split("");  // 按空字符串分割
        String[] words2 = text2.split("");  // 按空字符串分割

        // 加载词汇表
        Map<String, Integer> vocab = null;
        try {
            vocab = VocabLoader.loadVocab(this, "hanlp/vocab.txt");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // 转换为模型输入格式
        int maxLength = 128; // 最大长度，依据您的需求进行调整

        long[][] input1 = convertToIds(words1, vocab, maxLength);
        long[][] input2 = convertToIds(words2, vocab, maxLength);

        begin(input1,input2);

    }

}