/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model.Anchor;

import Model.Embedding.Embedding;
import Model.Embedding.EmbeddingDouble;
import Model.Range;
import Word2VecParser.*;
import static Word2VecParser.Word2VecParser.NUMBER_RANGES;
import static Word2VecParser.Word2VecParser.SIMILARITY_FUNCTION;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

/**
 *
 * @author msobroza
 */
public class AnchorDistribution extends Anchor {
    

    public AnchorDistribution(EmbeddingDouble e, List<EmbeddingDouble> vectors) {
        super(e);
        double dAux;
        this.MIN_DISTANCE_DISTRIBUTION = Double.POSITIVE_INFINITY;
        this.MAX_DISTANCE_DISTRIBUTION = Double.NEGATIVE_INFINITY;
        // Creates distance map and interval
        for (EmbeddingDouble aux : vectors) {
            dAux = calculateDistance(this, aux);
            if (dAux < MIN_DISTANCE_DISTRIBUTION) {
                MIN_DISTANCE_DISTRIBUTION = dAux;
            }
            if (dAux > MAX_DISTANCE_DISTRIBUTION) {
                MAX_DISTANCE_DISTRIBUTION = dAux;
            }
            this.embMapDist.put(aux, dAux);
        }
        if (SIMILARITY_FUNCTION.getIndex() == Config.SimilarityFunction.CosinusDistance.getIndex() && MIN_DISTANCE_DISTRIBUTION < 0.0) {
            for (EmbeddingDouble aux : vectors) {
                this.embMapDist.put(aux, this.embMapDist.get(aux) - MIN_DISTANCE_DISTRIBUTION);
            }
            MAX_DISTANCE_DISTRIBUTION = MAX_DISTANCE_DISTRIBUTION - MIN_DISTANCE_DISTRIBUTION;
            MIN_DISTANCE_DISTRIBUTION = 0.0;
        }

        RANGE_DISTANCE_DISTRIBUTION = MAX_DISTANCE_DISTRIBUTION - MIN_DISTANCE_DISTRIBUTION;
        interval = RANGE_DISTANCE_DISTRIBUTION / NUMBER_RANGES;
        interval_elements = vectors.size() / NUMBER_RANGES;
        // Creates ranges
        if (Word2VecParser.FIXED_DISTANCE_INTERVAL_RANGES) {

            for (int i = 0; i < NUMBER_RANGES; i++) {
                addRange(i, MIN_DISTANCE_DISTRIBUTION + i * interval, MIN_DISTANCE_DISTRIBUTION + (i + 1) * interval);
            }
            for (EmbeddingDouble v : vectors) {
                Double distanceV = this.embMapDist.get(v);
                Range rangeV = getRange(distanceV);
                if (!rangeV.isInRange(v)) {
                    rangeV.addVector(v);
                }
            }
        } else {
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
                    numberElementsRange = 0;
                    rangeEmbeddings = new HashSet<>();
                    countRange++;
                }
            }
        }

    }

    public AnchorDistribution(String word, double[] vector, List<EmbeddingDouble> vectors) {
        super(word, vector);
        double dAux;
        this.MIN_DISTANCE_DISTRIBUTION = Double.POSITIVE_INFINITY;
        this.MAX_DISTANCE_DISTRIBUTION = Double.NEGATIVE_INFINITY;
        // Creates distance map and interval
        for (EmbeddingDouble aux : vectors) {
            dAux = calculateDistance(this, aux);
            if (dAux < MIN_DISTANCE_DISTRIBUTION) {
                MIN_DISTANCE_DISTRIBUTION = dAux;
            }
            if (dAux > MAX_DISTANCE_DISTRIBUTION) {
                MAX_DISTANCE_DISTRIBUTION = dAux;
            }
            this.embMapDist.put(aux, dAux);
        }
        if (SIMILARITY_FUNCTION.getIndex() == Config.SimilarityFunction.CosinusDistance.getIndex() && MIN_DISTANCE_DISTRIBUTION < 0.0) {
            for (EmbeddingDouble aux : vectors) {
                this.embMapDist.put(aux, this.embMapDist.get(aux) - MIN_DISTANCE_DISTRIBUTION);
            }
        }

        RANGE_DISTANCE_DISTRIBUTION = MAX_DISTANCE_DISTRIBUTION - MIN_DISTANCE_DISTRIBUTION;
        interval = RANGE_DISTANCE_DISTRIBUTION / NUMBER_RANGES;
        interval_elements = vectors.size() / NUMBER_RANGES;
        // Creates ranges
        if (Word2VecParser.FIXED_DISTANCE_INTERVAL_RANGES) {

            for (int i = 0; i < NUMBER_RANGES; i++) {
                addRange(i, MIN_DISTANCE_DISTRIBUTION + i * interval, MIN_DISTANCE_DISTRIBUTION + (i + 1) * interval);
            }
            for (EmbeddingDouble v : vectors) {
                Double distanceV = this.embMapDist.get(v);
                Range rangeV = getRange(distanceV);
                if (!rangeV.isInRange(v)) {
                    rangeV.addVector(v);
                }
            }
        } else {
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
                    numberElementsRange = 0;
                    rangeEmbeddings = new HashSet<>();
                    countRange++;
                }
            }
        }
    }

}
