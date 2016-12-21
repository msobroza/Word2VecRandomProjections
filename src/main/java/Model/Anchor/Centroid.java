/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model.Anchor;

import Model.Embedding.EmbeddingDouble;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author msobroza
 */
public class Centroid extends EmbeddingDouble {
    private final int idCentroid;

    public Centroid(int idCentroid, INDArray vector) {
        super(idCentroid + "", vector);
        this.idCentroid=idCentroid;
    }
    
    public int getIdCentroid(){
        return this.idCentroid;
    }
}
