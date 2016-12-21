/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model.Anchor;

import Model.Anchor.Anchor;
import Model.Embedding.Embedding;
import Model.Embedding.EmbeddingDouble;
import Word2VecParser.Config;
import Word2VecParser.Word2VecParser;
import static Word2VecParser.Word2VecParser.NUMBER_RANGES;
import static Word2VecParser.Word2VecParser.SIMILARITY_FUNCTION;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author msobroza
 */
public class AnchorRandomCode extends Anchor {

    private static final double[] codeWords = {0.0, 1.0};
    private final int numCode;
    private double sumIntervals;

    public AnchorRandomCode(int numCode, int length, List<EmbeddingDouble> vectors) {
        super();
        this.numCode = numCode;
        this.vector = Nd4j.create(generateRandomCode(length));
        double dAux;
        this.ranges = new ArrayList<>();
        this.embMapDist = new HashMap<>();
        this.MIN_DISTANCE_DISTRIBUTION = Double.POSITIVE_INFINITY;
        this.MAX_DISTANCE_DISTRIBUTION = Double.NEGATIVE_INFINITY;
        this.sumIntervals = 0.0;
        // Creates distance map and interval
        for (EmbeddingDouble aux : vectors) {
            dAux = dotProduct(this, aux);
            if (dAux < MIN_DISTANCE_DISTRIBUTION) {
                MIN_DISTANCE_DISTRIBUTION = dAux;
            }
            if (dAux > MAX_DISTANCE_DISTRIBUTION) {
                MAX_DISTANCE_DISTRIBUTION = dAux;
            }
            this.embMapDist.put(aux, dAux);
        }
        if (MIN_DISTANCE_DISTRIBUTION < 0.0) {
            for (EmbeddingDouble aux : vectors) {
                this.embMapDist.put(aux, this.embMapDist.get(aux) - MIN_DISTANCE_DISTRIBUTION);
            }
        }

        RANGE_DISTANCE_DISTRIBUTION = MAX_DISTANCE_DISTRIBUTION - MIN_DISTANCE_DISTRIBUTION;
        interval = RANGE_DISTANCE_DISTRIBUTION / NUMBER_RANGES;
        interval_elements = vectors.size() / NUMBER_RANGES;

        int countRange = 0;
        ArrayList<EmbeddingDouble> sortedEmbeddings = new ArrayList<>();
        sortedEmbeddings.addAll(embMapDist.keySet());
        Collections.sort(sortedEmbeddings, new CompareDistanceFromAnchor());
        int numberElementsRange = 0;
        HashSet<Embedding> rangeEmbeddings = new HashSet<>();
        Double minRangeDistance = embMapDist.get(sortedEmbeddings.get(0));
        for (int i = 0; i < sortedEmbeddings.size(); i++) {
            if (numberElementsRange < interval_elements || Objects.equals(embMapDist.get(sortedEmbeddings.get(i - 1)), embMapDist.get(sortedEmbeddings.get(i)))) {
                rangeEmbeddings.add(sortedEmbeddings.get(i));
                if (numberElementsRange == 0) {
                    minRangeDistance = embMapDist.get(sortedEmbeddings.get(i));
                }
                numberElementsRange++;
            }
            if (i + 1 == sortedEmbeddings.size() || numberElementsRange >= interval_elements && (i + 1 < sortedEmbeddings.size() && !Objects.equals(embMapDist.get(sortedEmbeddings.get(i)), embMapDist.get(sortedEmbeddings.get(i + 1))))) {
                Double maxRangeDistance = embMapDist.get(sortedEmbeddings.get(i));
                System.out.println("Creating range: " + countRange + " , MinDistance: " + minRangeDistance + " , MaxDistance: " + maxRangeDistance + " NumberElements: " + numberElementsRange);
                addRange(countRange, minRangeDistance, maxRangeDistance, rangeEmbeddings);
                sumIntervals += (maxRangeDistance - minRangeDistance);
                numberElementsRange = 0;
                rangeEmbeddings = new HashSet<>();
                countRange++;
            }
        }

    }

    public double getSumIntervals() {
        return this.sumIntervals;
    }

    public int getNumCode() {
        return this.numCode;
    }

    protected static double dotProduct(EmbeddingDouble a, EmbeddingDouble b) {
        return Nd4j.getBlasWrapper().dot((INDArray) a.getVector(), (INDArray) b.getVector());
    }

    private static double[] generateRandomCode(int length) {
        Random rdGen = new Random();
        double code[] = new double[length];
        for (int i = 0; i < length; i++) {
            code[i] = codeWords[rdGen.nextInt(2)];
        }
        return code;
    }

}
