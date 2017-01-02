/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Word2VecParser;

import Model.Embedding.EmbeddingDouble;
import RandomProjectionNetwork.RPDenseLayer;
import RandomProjectionNetwork.DifferentialLayer.RPDifferentialHardLayer;
import RandomProjectionNetwork.DifferentialLayer.RPDifferentialSoftLayer;
import RandomProjectionNetwork.InputLayer.RPInputWordLayer;
import RandomProjectionNetwork.RPWordNetwork;
import Tools.SynonymDict;
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
    public static int FANALS_PER_CLUSTER = 2;
    public static int NUM_CLUSTERS = 2048;
    public static final int EMBEDDINGS_DIMENSIONS = 300;
    public static final String INPUT_EMBEDDINGS = "./gloveVectors/glove.6B." + EMBEDDINGS_DIMENSIONS + "d.txt";
    public static final String OUTPUT_FILE = "./code5_binary_glove.6B." + EMBEDDINGS_DIMENSIONS + "d.txt";
    public static final String INPUT_SYNONYM = "./Synonyms/syn-glove.6B.300d.txt";
    public static final boolean STM_ENABLED = true;
    public static final int NUMBER_SAMPLES_PER_PATTERN = 3;
    public static final Double THRESHOLD_STM = 0.1;
    public static final boolean DIFFERENTIAL_LAYER = true;
    public static final boolean SOFT_DECISION_ENABLED = true;

    public static void main(String[] args) throws IOException, Exception {
        Word2VecParserRandom.log.info("Random weights matrix Claude idea");
        RPWordNetwork net;
        if (STM_ENABLED) {
            net = new RPWordNetwork(INPUT_EMBEDDINGS, INPUT_SYNONYM, EMBEDDINGS_DIMENSIONS, NUMBER_SAMPLES_PER_PATTERN);
        } else {
            net = new RPWordNetwork(INPUT_EMBEDDINGS, EMBEDDINGS_DIMENSIONS);
        }
        if (DIFFERENTIAL_LAYER) {
            if (SOFT_DECISION_ENABLED) {
                net.addLayer(new RPDifferentialSoftLayer(EMBEDDINGS_DIMENSIONS, NUM_CLUSTERS, FANALS_PER_CLUSTER, net.getInputLayer().getNumberWords(), (float) -1.0, (float) 1.0));
            } else {
                net.addLayer(new RPDifferentialHardLayer(EMBEDDINGS_DIMENSIONS, NUM_CLUSTERS, FANALS_PER_CLUSTER, net.getInputLayer().getNumberWords(), (float) -1.0, (float) 1.0));
            }
        } else {
            net.addLayer(new RPDenseLayer(EMBEDDINGS_DIMENSIONS, NUM_CLUSTERS, FANALS_PER_CLUSTER, (float) -1.0, (float) 1.0));
        }
        int j = 0;
        RPInputWordLayer inputLayer = net.getInputLayer();
        ArrayList<EmbeddingDouble> vectorsBinary = new ArrayList<>();
        for (int i = 0; i < inputLayer.getNumberWords(); i++) {
            String word = inputLayer.getWord(i);
            if (DIFFERENTIAL_LAYER && STM_ENABLED) {
                if (SOFT_DECISION_ENABLED) {
                    vectorsBinary.add(new EmbeddingDouble(word, net.getCliqueOutputSoftSTM(word, THRESHOLD_STM)));
                } else {
                    vectorsBinary.add(new EmbeddingDouble(word, net.getCliqueOutputHardSTM(word, THRESHOLD_STM)));
                }
            } else {
                vectorsBinary.add(new EmbeddingDouble(word, net.getCliqueOutput(word)));
            }
            if (j % 10000 == 0) {
                Word2VecParserRandom.log.info("j= " + j);
            }
            j++;
        }
        if (DIFFERENTIAL_LAYER && STM_ENABLED) {
            System.out.println("Original vectors: " + net.getNumberOriginalVectors());
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
