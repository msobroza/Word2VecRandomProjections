/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Word2VecParser;

import Tools.KMeans;
import CliqueNetwork.CliqueNet;
import static Tools.BinaryOperations.binarizeVectorToBitSet;
import static Tools.BinaryOperations.binarizeVectorToDoubleArray;
import Model.Embedding.Embedding;
import Model.Embedding.EmbeddingBit;
import Model.Embedding.EmbeddingDouble;
import Model.Embedding.EmbeddingHadamard;
import Model.Embedding.EmbeddingHadamardBit;
import Model.Embedding.EmbeddingReduce;
import Word2VecParser.Config.BinaryType;
import Word2VecParser.Config.MethodAnchorPoints;
import Word2VecParser.Config.SimilarityFunction;
import Tools.KMeans.DistanceFunction;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author msobroza
 */
public class Word2VecParser {

    public static int NUMBER_RANGES = 250;
    public static int NUMBER_ANCHOR_POINTS = 8;
    public static boolean ACTIVATE_FUZZY = true;
    public static int FUZZY_PARAMETER = 10;
    public static int FANALS_PER_CLUSTER = 10;
    public static int GAMA = 1;
    public static int ACTIVATION_VALUE = 1;
    public static SimilarityFunction SIMILARITY_FUNCTION = SimilarityFunction.HammingDistance;
    public static MethodAnchorPoints METHOD_ANCHOR_POINTS = MethodAnchorPoints.DistributedHadarmardCode;
    public static boolean FIXED_DISTANCE_INTERVAL_RANGES = true;
    // Input vectors options
    public static boolean NORMALIZE_INPUT_VECTORS = false;
    public static boolean BINARIZE_INPUT_VECTORS = false;
    public static double BINARY_THRESHOLD = 0.0;
    public static BinaryType TYPE_BINARY_INPUT_VECTORS = BinaryType.DoubleType;

    // K-means parameters
    public static int KMEANS_MAX_ITERATION_COUNT = 4;
    public static KMeans.DistanceFunction KMEANS_DISTANCE_FUNCTION = DistanceFunction.SquaredDistance;

    // Random binaries codes parameters
    public static int NUMBER_BINARY_CODES = 15;
    public static int NUMBER_BINARY_SELECTED_CODES = NUMBER_ANCHOR_POINTS;
    public static int CODE_LENGTH = 50;

    // Distribution of Hadamard codes
    public static int NUMBER_DISTRIBUTIONS = 1;
    public static int N_HADAMARD = 1024;

    // Test parameters
    public static int NUMBER_NEAREST_WORDS = 1;
    public static final Logger log = LoggerFactory.getLogger(Word2VecParser.class);

    public static void main(String[] args) throws IOException, Exception {
        ArrayList<String> resultDecoder;
        WordVectors vec = WordVectorSerializer.loadTxtVectors(new File("./glove/binary_glove.6B.100d_1024d.txt"));
        //WordVectors vec2 = WordVectorSerializer.loadTxtVectors(new File("./glove.6B.100d.txt"));
        HashMap<String, Embedding> vectorMap = new HashMap<>();
        int numWords = vec.vocab().numWords();
        Word2VecParser.log.info("Num of words: " + numWords);
        Word2VecParser.log.info("Getting embedding representation...");
        ArrayList<EmbeddingDouble> vectorsDouble = new ArrayList<>();
        ArrayList<EmbeddingBit> vectorsBit = new ArrayList<>();

        if (NORMALIZE_INPUT_VECTORS) {
            for (int i = 0; i < numWords; i++) {
                String word = vec.vocab().wordAtIndex(i);
                EmbeddingDouble e = new EmbeddingDouble(word, vec.getWordVectorMatrixNormalized(word));
                vectorsDouble.add(e);
                vectorMap.put(word, e);
            }
        } else {
            if (BINARIZE_INPUT_VECTORS) {
                if (TYPE_BINARY_INPUT_VECTORS.getIndex() == BinaryType.DoubleType.getIndex()) {
                    for (int i = 0; i < numWords; i++) {
                        String word = vec.vocab().wordAtIndex(i);
                        EmbeddingDouble e = new EmbeddingDouble(word, binarizeVectorToDoubleArray(vec.getWordVector(word), BINARY_THRESHOLD));
                        vectorsDouble.add(e);
                        vectorMap.put(word, e);
                    }
                } else {
                    for (int i = 0; i < numWords; i++) {
                        String word = vec.vocab().wordAtIndex(i);
                        EmbeddingBit e = new EmbeddingBit(word, binarizeVectorToBitSet(vec.getWordVector(word), BINARY_THRESHOLD));
                        vectorsBit.add(e);
                        vectorMap.put(word, e);
                    }
                }
            } else {
                for (int i = 0; i < numWords; i++) {
                    String word = vec.vocab().wordAtIndex(i);
                    EmbeddingDouble e = new EmbeddingDouble(word, vec.getWordVector(word));
                    vectorsDouble.add(e);
                    vectorMap.put(word, e);
                }
            }

        }
        Word2VecParser.log.info("Choosing reference points and calculating distances...");
        CliqueNet net = new CliqueNet();
        if (METHOD_ANCHOR_POINTS.getIndex() == MethodAnchorPoints.DistributedHadarmardCode.getIndex()) {
            Word2VecParser.log.info("Distributed hadamard code");
            // net.createDistributedHadamardCodes(vectorsBit);
        } else if (METHOD_ANCHOR_POINTS.getIndex() == MethodAnchorPoints.Random.getIndex()) {
            Word2VecParser.log.info("Random points method");
            net.createRandomAnchorPoints(vectorsDouble);
        } else if (METHOD_ANCHOR_POINTS.getIndex() == MethodAnchorPoints.KMeansCentroids.getIndex()) {
            Word2VecParser.log.info("KMeans centroids method");
            net.createKMeansCentroidsAnchorPoints(vectorsDouble);
        } else if (METHOD_ANCHOR_POINTS.getIndex() == MethodAnchorPoints.RandomBinaryCodes.getIndex()) {
            Word2VecParser.log.info("Random binary codes method");
            net.createRandomBinaryCodes(vectorsDouble);
        } else {
            ArrayList<EmbeddingReduce> vectorsReduce = new ArrayList<>();
            for (Embedding v : vectorsDouble) {
                vectorsReduce.add(new EmbeddingReduce((EmbeddingDouble) v));
            }
            Word2VecParser.log.info("Sum method");
            net.createSumBasedAnchorPoints(vectorsReduce);
        }
        Word2VecParser.log.info("Decoding word...");
        ArrayList<String> resultNearestNeighbors;
        Double totalAccuracy = 0.0;
        int nFound = 0;
        int n = 0;
        Collections.shuffle(vectorsDouble);
        Collections.shuffle(vectorsBit);
        if (METHOD_ANCHOR_POINTS.getIndex() == MethodAnchorPoints.DistributedHadarmardCode.getIndex()) {
            net.decoderFuzzyCodeHadamard((EmbeddingHadamardBit) vectorMap.get("day"), FUZZY_PARAMETER);
        } else if (ACTIVATE_FUZZY) {
            System.out.println("Fuzzy: " + FUZZY_PARAMETER);
            resultDecoder = net.decoderFuzzyAnchorPoint(vectorMap.get("day"), FUZZY_PARAMETER);
            resultNearestNeighbors = (ArrayList<String>) vec.wordsNearest("day", 5);
            for (String s : resultDecoder) {
                System.out.println("Fist decoding output: <" + s + ">");
            }
            System.out.println(resultNearestNeighbors);
            for (EmbeddingDouble eVocab : vectorsDouble) {
                resultDecoder = net.decoderFuzzyAnchorPoint(eVocab, FUZZY_PARAMETER);
                resultNearestNeighbors = (ArrayList<String>) vec.wordsNearest(eVocab.getWord(), NUMBER_NEAREST_WORDS);
                //resultNearestNeighbors.remove(0);
                totalAccuracy += getAccuracyWord(resultDecoder, resultNearestNeighbors);
                nFound += resultDecoder.size();
                n = n + 1;
                if (n % 100 == 0) {
                    System.out.println("Number of tested words: " + n);
                    System.out.println("Partial accuracy: " + totalAccuracy / n);
                    System.out.println("Partial average answers:" + ((double) nFound) / n);
                }
            }
        } else {
            for (EmbeddingDouble eVocab : vectorsDouble) {
                resultDecoder = net.decoderAnchorPoint(eVocab);
                resultNearestNeighbors = (ArrayList<String>) vec.wordsNearest(eVocab.getWord(), NUMBER_NEAREST_WORDS);
                totalAccuracy += getAccuracyWord(resultDecoder, resultNearestNeighbors);
                nFound += resultDecoder.size();
                n = n + 1;
                if (n % 100 == 0) {
                    System.out.println("Number of tested words: " + n);
                    System.out.println("Partial accuracy: " + totalAccuracy / n);
                    System.out.println("Partial average answers:" + ((double) nFound) / n);
                }

            }
        }
        System.out.println("Average results per decoding: " + ((double) nFound) / n);
        System.out.println("Average accuracy network: " + (totalAccuracy / n));

        /*UiServer server = UiServer.getInstance();
         System.out.println("Started on port " + server.getPort());*/
    }

    public static Double getAccuracyWord(ArrayList<String> predictWords, ArrayList<String> correctWords) {
        int totalWords = correctWords.size();
        int numberCorrects = 0;
        for (String w : correctWords) {
            if (predictWords.contains(w)) {
                numberCorrects++;
            }
        }
        return ((double) numberCorrects) / totalWords;
    }
}
