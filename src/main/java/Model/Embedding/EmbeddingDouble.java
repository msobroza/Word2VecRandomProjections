/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model.Embedding;

import Model.Embedding.Embedding;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 *
 * @author msobroza
 */
public class EmbeddingDouble extends Embedding {

    protected INDArray vector;

    
    public EmbeddingDouble(){
    }
    
    
    public EmbeddingDouble(String word, double[] vector) {
        super(word);
        this.vector = Nd4j.create(vector);
    }

    public EmbeddingDouble(String word, INDArray vector) {
        super(word);
        this.vector = vector;
    }

    public EmbeddingDouble(EmbeddingDouble e) {
        super(e.getWord());
        this.vector = e.vector;
    }

    @Override
    public Object getVector() {
        return this.vector;
    }

    public double getCosinusDistance(INDArray vectorB) {
        return Transforms.cosineSim(vector, vectorB);
    }

    public double getL2Distance(INDArray vectorB) {
        return vector.distance2(vectorB);
    }

    public double getSquaredDistance(INDArray vectorB) {
        return vector.squaredDistance(vectorB);
    }

    public int getHammingDistance(INDArray binaryVectorB) {
        int dimensions = this.vector.columns();
        if (dimensions != binaryVectorB.columns()) {
            return -1;
        }
        int count = 0;
        for (int i = 0; i < dimensions; i++) {
            if (this.vector.getDouble(i) == binaryVectorB.getDouble(i)) {
                count++;
            }
        }
        return dimensions - count;
    }

}
