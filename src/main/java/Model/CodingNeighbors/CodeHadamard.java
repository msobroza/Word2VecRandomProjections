/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model.CodingNeighbors;

import Model.Embedding.EmbeddingDouble;
import java.util.BitSet;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author msobroza
 */
public class CodeHadamard extends EmbeddingDouble {

    private final int numCode;

    public CodeHadamard(int numCode, EmbeddingDouble e) {
        super(e);
        this.numCode = numCode;
    }

    public CodeHadamard(int numCode, INDArray vector) {
        super(numCode + "", vector);
        this.numCode = numCode;
    }

    public int getNumCode() {
        return this.numCode;
    }

}
