/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model.Embedding;

import java.util.BitSet;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author msobroza
 */
public class EmbeddingBit extends Embedding {

    protected BitSet vector;

    public EmbeddingBit(String word, BitSet vector) {
        super(word);
        this.vector = vector;
    }
    
    public EmbeddingBit(BitSet vector) {
        this.vector = vector;
    }

    public EmbeddingBit(EmbeddingBit e) {
        this.vector = e.vector;
    }

    public int getHammmingDistance(BitSet binaryVectorB) {
        BitSet resultXor = (BitSet) vector.clone();
        resultXor.xor(binaryVectorB);
        return resultXor.cardinality();
    }

    @Override
    public Object getVector() {
        return vector;
    }

}
