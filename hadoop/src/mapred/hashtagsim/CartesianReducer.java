package mapred.hashtagsim;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.*;

/**
 * Created by LumiG on 3/15/16.
 */
public class CartesianReducer extends Reducer<Text, Text, Text, Text> {
    @Override
    protected void reduce(Text key, Iterable<Text> value,
                          Context context)
            throws IOException, InterruptedException {

        List<String> valueSet = new ArrayList<>();
        for (Text word : value) {
//            System.out.println("################################");
//            System.out.println("################################");
//            System.out.println(word.toString());
//            System.out.println("################################");
//            System.out.println("################################");
            valueSet.add(word.toString());
        }

//        System.out.println("################################");
//        System.out.println("################################");
//        System.out.println(valueSet);
//        System.out.println("################################");
//        System.out.println("################################");

        for(int i = 0; i < valueSet.size(); i++){
            for(int j = i+1; j < valueSet.size(); j++){
                StringBuilder builder = new StringBuilder();
                builder.append(valueSet.get(i));
                builder.append("|");
                builder.append(valueSet.get(j));
                context.write(new Text("key"), new Text(builder.toString()));
            }
        }

    }
}
