/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model.CodingNeighbors;

import Model.Anchor.Anchor;
import Model.Embedding.Embedding;
import Model.Embedding.EmbeddingBit;
import Model.Embedding.EmbeddingHadamard;
import Model.Embedding.EmbeddingHadamardBit;
import Model.Range;
import static Word2VecParser.Word2VecParser.NUMBER_RANGES;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

/**
 *
 * @author msobroza
 */
public class CodeDistribution extends Embedding{

    private final ArrayList<CodeHadamardBit> codes;
    private final HashMap<EmbeddingHadamardBit, CodeHadamardBit> embeddingMapCode;
    private final HashMap<CodeHadamardBit, Range> codeMapRange;
    private final ArrayList<Range> ranges;

    public CodeDistribution(ArrayList<CodeHadamardBit> codes) {
        this.codes = codes;
        this.codeMapRange = new HashMap<>();
        this.embeddingMapCode = new HashMap<>();
        this.ranges = new ArrayList<>();
        // Initializes one range for each hadamard's code
        for (CodeHadamardBit c : codes) {
            Range r = new Range(c.getNumCode());
            this.codeMapRange.put(c, r);
            this.ranges.add(r);
        }
    }

    public void addEmbedding(EmbeddingHadamardBit eh, int pos) {
        if (pos < 0 || pos >= codes.size()) {
            return;
        }
        CodeHadamardBit c = eh.getCodeHadamardRank(pos);
        if (!codes.contains(c)) {
            return;
        }
        // Adds embeddings in range
        Range r = codeMapRange.get(c);
        r.addVector(eh);
        // Create references from embedding to a code
        this.embeddingMapCode.put(eh, c);
    }

    public Range getRange(CodeHadamardBit c) {
        return this.codeMapRange.get(c);

    }

    public HashSet<Range> getFuzzyRange(EmbeddingBit e, int nFuzzy) {
        HashSet<Range> result = new HashSet<>();
        EmbeddingHadamardBit eh = (EmbeddingHadamardBit) e;
        CodeHadamardBit c = this.embeddingMapCode.get(eh);
        int idRank = eh.getRank(c);
        int idWin;
        for (int i = -nFuzzy; i <= nFuzzy; i++) {
            idWin = idRank + i;
            if (idWin < 0 || idWin >= this.ranges.size()) {
                continue;
            }
            result.add(this.codeMapRange.get(eh.getCodeHadamardRank(idWin)));
        }
        return result;
    }

    public Range getRange(EmbeddingBit e) {
        return this.codeMapRange.get(this.embeddingMapCode.get((EmbeddingHadamardBit) e));
    }

    public static int calculateDistance(EmbeddingBit binaryVectorA, EmbeddingBit binaryVectorB) {
        return binaryVectorA.getHammmingDistance((BitSet) binaryVectorB.getVector());
    }

    @Override
    public Object getVector() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
