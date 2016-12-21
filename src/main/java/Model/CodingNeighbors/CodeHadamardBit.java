/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model.CodingNeighbors;

import Model.Embedding.EmbeddingBit;
import java.util.BitSet;

/**
 *
 * @author msobroza
 */
public class CodeHadamardBit extends EmbeddingBit {

    private int numCode;

    public CodeHadamardBit(int numCode, BitSet binaryVector) {
        super(binaryVector);
        this.numCode = numCode;
    }
    
    public int getNumCode(){
        return this.numCode;
    }
}
