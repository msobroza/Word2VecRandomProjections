/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package RandomProjectionNetwork;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author msobroza
 */
public class RPWordNetwork {

    private RPInputWordLayer inputLayer;
    private ArrayList<RPDenseLayer> layers;

    public RPWordNetwork(String fileInput, int embeddingsDimension) throws FileNotFoundException, UnsupportedEncodingException {
        this.inputLayer = new RPInputWordLayer(fileInput, embeddingsDimension);
    }

    public INDArray getCliqueOutput(String word) {
        INDArray inputVector = inputLayer.getWordVector(word);
        int numLayers = 0;
        for (RPDenseLayer l : layers) {
            inputVector = l.getActivationsMult(inputVector);
            if (numLayers == 0 && l instanceof RPDifferentialLayer) {
                inputVector = ((RPDifferentialLayer) l).getVectorDifferentialWTA(inputVector);
            } else {
                inputVector = l.getVectorFromCliques(l.getCliqueIndexesWTA(inputVector));
            }
        }
        return inputVector;
    }

    public void addLayer(RPDenseLayer layer) {
        this.layers.add(layer);
    }

    public RPDenseLayer getLayer(int index) {
        return this.layers.get(index);
    }

    public RPInputWordLayer getInputLayer() {
        return this.inputLayer;
    }
}
