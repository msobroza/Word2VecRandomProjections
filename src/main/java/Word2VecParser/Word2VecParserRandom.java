/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Word2VecParser;

import CliqueNetwork.CliqueNet;
import Model.Embedding.Embedding;
import Model.Embedding.EmbeddingDouble;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author msobroza
 */
public class Word2VecParserRandom {

    public static final Logger log = LoggerFactory.getLogger(Word2VecParserRandom.class);
    public static int FANALS_PER_CLUSTER = 128;
    public static int NUM_CLUSTERS = 32;
    public static final int EMBEDDINGS_DIMENSIONS = 300;
    public static final String OUTPUT_FILE= "./code1_binary_glove.6B." + EMBEDDINGS_DIMENSIONS + "d.txt";

    public static void main(String[] args) throws IOException, Exception {
        WordVectors vec = WordVectorSerializer.loadTxtVectors(new File("./gloveVectors/glove.6B." + EMBEDDINGS_DIMENSIONS + "d.txt"));
        int numWords = vec.vocab().numWords();
        Word2VecParser.log.info("Num of words: " + numWords);
        HashMap<String, Integer> vectorMap = new HashMap<>();
        ArrayList<EmbeddingDouble> vectorsBinary = new ArrayList<>();
        for (int i = 0; i < numWords; i++) {
            String word = vec.vocab().wordAtIndex(i);
            vectorMap.put(word, i);
        }
        Word2VecParserRandom.log.info("Random weights matrix Claude idea");
        CliqueNet net = new CliqueNet();
        net.createRandomMatrixInitialization(EMBEDDINGS_DIMENSIONS, FANALS_PER_CLUSTER, NUM_CLUSTERS, (float) -1.0, (float) 1.0);
        int j=0;
        INDArray v;
        int[] indexVec;
        for (int i=0; i< numWords; i++) {
            String word= vec.vocab().wordAtIndex(i);
            indexVec = net.getCliqueFromRandomMultiplicationMatrix(word, vec.getWordVectorMatrix(word));
            v = Nd4j.zeros(FANALS_PER_CLUSTER * NUM_CLUSTERS);
            for (int iCluster = 0; iCluster < NUM_CLUSTERS; iCluster++) {
                v.putScalar(iCluster * FANALS_PER_CLUSTER + indexVec[iCluster], 1.0);
            }
            vectorsBinary.add(new EmbeddingDouble(vec.vocab().wordAtIndex(i), v));
            if(j%10000==0){
                Word2VecParserRandom.log.info("j= "+j);
            }
            j++;
        }
        writeWordVectors(vectorsBinary,OUTPUT_FILE);
    }

    public static void writeWordVectors(ArrayList<EmbeddingDouble> wordVectors, String path) throws IOException {
        BufferedWriter write = new BufferedWriter(new FileWriter(new File(path), false));
        int words = 0;
        for (int i = 0; i < wordVectors.size(); i++) {
            String word = wordVectors.get(i).getWord();
            if (word == null) {
                continue;
            }
            StringBuffer sb = new StringBuffer();
            sb.append(word);
            sb.append(" ");
            INDArray wordVector = (INDArray) (wordVectors.get(i).getVector());
            for (int j = 0; j < wordVector.length(); j++) {
                sb.append(wordVector.getDouble(j));
                if (j < wordVector.length() - 1) {
                    sb.append(" ");
                }
            }
            sb.append("\n");
            write.write(sb.toString());

        }

        write.flush();
        write.close();

    }
}
