/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model.Embedding;

import Model.Embedding.EmbeddingDouble;

/**
 *
 * @author msobroza
 */
public class EmbeddingReduce extends EmbeddingDouble {

    private final Double moduloSum;
    private final Double l1Norm0;

    public EmbeddingReduce(String word, double[] vector) {
        super(word, vector);
        this.moduloSum = Math.abs(this.vector.sumNumber().doubleValue());
        this.l1Norm0 = this.vector.norm1Number().doubleValue();
    }

    public EmbeddingReduce(EmbeddingDouble e) {
        super(e);
        this.moduloSum = Math.abs(vector.sumNumber().doubleValue());
        this.l1Norm0 = vector.norm1Number().doubleValue();
    }
    
    public Double getAbsSumComponents() {
        return this.moduloSum;
    }
    
    public Double getL1Norm0(){
        return this.l1Norm0;
    }
    

}
