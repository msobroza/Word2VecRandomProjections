/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model.Anchor;

import Model.Embedding.Embedding;
import Model.Embedding.EmbeddingDouble;
import Model.Range;
import Word2VecParser.Config;
import static Word2VecParser.Word2VecParser.FIXED_DISTANCE_INTERVAL_RANGES;
import static Word2VecParser.Word2VecParser.NUMBER_RANGES;
import static Word2VecParser.Word2VecParser.SIMILARITY_FUNCTION;
import static Word2VecParser.Word2VecParser.TYPE_BINARY_INPUT_VECTORS;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Objects;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author msobroza
 */
public class Anchor extends EmbeddingDouble {

    protected Double MIN_DISTANCE_DISTRIBUTION;
    protected Double MAX_DISTANCE_DISTRIBUTION;
    protected Double RANGE_DISTANCE_DISTRIBUTION;
    protected Double interval;
    protected int interval_elements;
    protected HashMap<EmbeddingDouble, Double> embMapDist;
    protected ArrayList<Range> ranges;

    class CompareDistanceFromAnchor implements Comparator<EmbeddingDouble> {

        @Override
        public int compare(EmbeddingDouble o1, EmbeddingDouble o2) {
            return (embMapDist.get(o1) < embMapDist.get(o2) ? -1 : (Objects.equals(embMapDist.get(o1), embMapDist.get(o2)) ? 0 : 1));
        }
    }
    
    public Anchor(){  
    }

    public Anchor(EmbeddingDouble e) {
        super(e);
        this.embMapDist = new HashMap<>();
        this.ranges = new ArrayList<>();
    }

    public Anchor(String word, double[] vector) {
        super(word, vector);
        this.embMapDist = new HashMap<>();
        this.ranges = new ArrayList<>();
    }

    public Range addRange(int idRange, Double minInterval, Double maxInterval) {
        Range result;
        result = new Range(idRange, minInterval, maxInterval);
        ranges.add(result);
        return result;
    }

    public Range addRange(int idRange, Double minInterval, Double maxInterval, HashSet<Embedding> emb) {
        Range result;
        result = new Range(idRange, minInterval, maxInterval, emb);
        ranges.add(result);
        return result;
    }

    public HashSet<Range> getFuzzyRange(Double value, int nFuzzy) {
        HashSet<Range> result = new HashSet<>();
        int idRange = getRange(value).getIdRange();
        int idWin;
        for (int i = -nFuzzy; i <= nFuzzy; i++) {
            idWin = idRange + i;
            if (idWin < 0 || idWin >= NUMBER_RANGES) {
                continue;
            }
            result.add(this.ranges.get(idWin));
        }
        return result;
    }

    public HashSet<Range> getFuzzyRange(Embedding e, int n) {
        return getFuzzyRange(embMapDist.get(e), n);
    }

    public Range getRange(int idRange) {
        return this.ranges.get(idRange);
    }

    public Range getRange(Double value) {
        int idRange;
        if (FIXED_DISTANCE_INTERVAL_RANGES) {
            idRange = (int) ((value - MIN_DISTANCE_DISTRIBUTION) / interval);
            if (idRange == NUMBER_RANGES) {
                idRange--;
            }
        } else {
            idRange = getRangeBinarySearch(0, NUMBER_RANGES - 1, value);
        }
        return getRange(idRange);
    }

    public int getRangeBinarySearch(int idMin, int idMax, Double value) {
        int idMed = ((int) ((idMax - idMin) / 2)) + idMin;
        //System.out.println("Call function getRangeBinarySearch(int idMin, int idMax, Double value): " + idMin + " , " + idMax + " , " + value + " idMed: " + idMed);
        if (value > getRange(idMed).getMaxValue()) {
            if (idMed == idMax - 1) {
                return getRangeBinarySearch(idMax, idMax, value);
                //return idMed;
            }
            return getRangeBinarySearch(idMed, idMax, value);
        } else if (value < getRange(idMed).getMinValue()) {
            return getRangeBinarySearch(idMin, idMed, value);
        } else {
            return idMed;
        }
    }

    public Range getRange(Embedding e) {
        return getRange(embMapDist.get(e));
    }

    public static double calculateDistance(EmbeddingDouble anchor, EmbeddingDouble point) {
        if (SIMILARITY_FUNCTION.getIndex() == Config.SimilarityFunction.CosinusDistance.getIndex()) {
            return anchor.getCosinusDistance((INDArray) point.getVector());
        } else if (SIMILARITY_FUNCTION.getIndex() == Config.SimilarityFunction.NormL2.getIndex()) {
            return anchor.getL2Distance((INDArray) point.getVector());
        } else if (SIMILARITY_FUNCTION.getIndex() == Config.SimilarityFunction.SquaredDistance.getIndex()) {
            return anchor.getSquaredDistance((INDArray) point.getVector());
        } else {
            return anchor.getHammingDistance((INDArray) point.getVector());
        }
    }
}
