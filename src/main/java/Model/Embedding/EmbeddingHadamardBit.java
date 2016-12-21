/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model.Embedding;

import static Model.Anchor.Anchor.calculateDistance;
import Model.CodingNeighbors.CodeDistribution;
import Model.CodingNeighbors.CodeHadamardBit;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Objects;

/**
 *
 * @author msobroza
 */
public class EmbeddingHadamardBit extends EmbeddingBit {

    private final ArrayList<CodeHadamardBit> codesRank;
    private final HashMap<CodeHadamardBit, Double> distanceCodeHadamard;

    class CompareDistanceCodes implements Comparator<CodeHadamardBit> {

        @Override
        public int compare(CodeHadamardBit o1, CodeHadamardBit o2) {
            return (distanceCodeHadamard.get(o1) < distanceCodeHadamard.get(o2) ? -1 : (Objects.equals(distanceCodeHadamard.get(o1), distanceCodeHadamard.get(o2)) ? 0 : 1));
        }
    }

    public EmbeddingHadamardBit(BitSet binaryVector, ArrayList<CodeHadamardBit> codes) {
        super(binaryVector);
        codesRank = new ArrayList<>(codes);
        distanceCodeHadamard = new HashMap<>();
    }

    public EmbeddingHadamardBit(EmbeddingBit e, ArrayList<CodeHadamardBit> codes) {
        super(e);
        codesRank = new ArrayList<>(codes);
        distanceCodeHadamard = new HashMap<>();
        for (CodeHadamardBit c : codesRank) {
            distanceCodeHadamard.put(c, (double) CodeDistribution.calculateDistance(c, e));
        }
        Collections.sort(codes, new CompareDistanceCodes());
    }

    public int getRank(CodeHadamardBit c) {
        return codesRank.indexOf(c);
    }

    public CodeHadamardBit getCodeHadamardRank(int pos) {
        return codesRank.get(pos);
    }

    public Double getDistanceFromCodeHadamard(int pos) {
        return distanceCodeHadamard.get(codesRank.get(pos));
    }

    public Double getDistanceFromCodeHadamard(CodeHadamardBit code) {
        return distanceCodeHadamard.get(code);
    }

}
