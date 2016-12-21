    /*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model.Embedding;

import Model.Embedding.EmbeddingDouble;
import Model.CodingNeighbors.CodeHadamard;
import static Model.Anchor.Anchor.calculateDistance;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Objects;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author msobroza
 */
public class EmbeddingHadamard extends EmbeddingDouble {

    private final ArrayList<CodeHadamard> codesRank;
    private final HashMap<CodeHadamard, Double> distanceCodeHadamard;

    class CompareDistanceCodes implements Comparator<CodeHadamard> {

        @Override
        public int compare(CodeHadamard o1, CodeHadamard o2) {
            return (distanceCodeHadamard.get(o1) < distanceCodeHadamard.get(o2) ? -1 : (Objects.equals(distanceCodeHadamard.get(o1), distanceCodeHadamard.get(o2)) ? 0 : 1));
        }
    }

    public EmbeddingHadamard(String word, INDArray vector, ArrayList<CodeHadamard> codes) {
        super(word, vector);
        codesRank = new ArrayList<>(codes);
        distanceCodeHadamard = new HashMap<>();
    }

    public EmbeddingHadamard(EmbeddingDouble e, ArrayList<CodeHadamard> codes) {
        super(e);
        codesRank = new ArrayList<>(codes);
        distanceCodeHadamard = new HashMap<>();
        for (CodeHadamard c : codesRank) {
            distanceCodeHadamard.put(c, calculateDistance(c, e));
        }
        Collections.sort(codes, new CompareDistanceCodes());
    }

    public int getRank(CodeHadamard c) {
        return codesRank.indexOf(c);
    }

    public CodeHadamard getCodeHadamardRank(int pos) {
        return codesRank.get(pos);
    }

    public Double getDistanceFromCodeHadamard(int pos) {
        return distanceCodeHadamard.get(codesRank.get(pos));
    }

    public Double getDistanceFromCodeHadamard(CodeHadamard code) {
        return distanceCodeHadamard.get(code);
    }
}
