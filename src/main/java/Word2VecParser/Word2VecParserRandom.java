/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Word2VecParser;

import Model.Embedding.EmbeddingDouble;
import RandomProjectionNetwork.RPDenseLayer;
import RandomProjectionNetwork.RPInputWordLayer;
import RandomProjectionNetwork.RPWordNetwork;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import org.nd4j.linalg.api.ndarray.INDArray;
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
    public static final String INPUT_FILE = "./gloveVectors/glove.6B." + EMBEDDINGS_DIMENSIONS + "d.txt";
    public static final String OUTPUT_FILE = "./code1_binary_glove.6B." + EMBEDDINGS_DIMENSIONS + "d.txt";
    public static final boolean DIFFERENTIAL_LAYER = false;

    public static void main(String[] args) throws IOException, Exception {
        Word2VecParserRandom.log.info("Random weights matrix Claude idea");
        RPWordNetwork net = new RPWordNetwork(INPUT_FILE, EMBEDDINGS_DIMENSIONS);
        RPInputWordLayer inputLayer = net.getInputLayer();
        net.addLayer(new RPDenseLayer(EMBEDDINGS_DIMENSIONS, NUM_CLUSTERS, FANALS_PER_CLUSTER, (float) -1.0, (float) 1.0));
        int j = 0;
        ArrayList<EmbeddingDouble> vectorsBinary = new ArrayList<>();
        for (int i = 0; i < inputLayer.getNumberWords(); i++) {
            String word = inputLayer.getWord(i);
            vectorsBinary.add(new EmbeddingDouble(word, net.getCliqueOutput(word)));
            if (j % 10000 == 0) {
                Word2VecParserRandom.log.info("j= " + j);
            }
            j++;
        }
        writeWordVectors(vectorsBinary, OUTPUT_FILE);
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
