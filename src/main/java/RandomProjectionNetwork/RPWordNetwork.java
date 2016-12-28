/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package RandomProjectionNetwork;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author msobroza
 */
public class RPWordNetwork {

    private RPInputWordLayer inputLayer;
    private ArrayList<RPDenseLayer> layers;
    private int numberOriginalVectors;

    public RPWordNetwork(String fileInput, int embeddingsDimension) throws FileNotFoundException, UnsupportedEncodingException {
        this.inputLayer = new RPInputWordLayer(fileInput, embeddingsDimension);
        this.layers = new ArrayList<>();
        this.numberOriginalVectors=0;
    }

    public RPWordNetwork(String fileNameEmbedding, String fileNameSyn, int wordEmbeddingsDimension, int minimumSamplesPerPattern) throws FileNotFoundException, UnsupportedEncodingException {
        this.inputLayer = new RPInputWordSynLayer(fileNameEmbedding, fileNameSyn, wordEmbeddingsDimension, minimumSamplesPerPattern);
        this.layers = new ArrayList<>();
        this.numberOriginalVectors=0;
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
            numLayers++;
        }
        return inputVector;
    }
    
    public int getNumberOriginalVectors(){
        return this.numberOriginalVectors;
    }

    public INDArray getCliqueOutputSTM(String word, Double thresholdSTM) {
        INDArray inputWordVector = inputLayer.getWordVector(word);
        INDArray inputVector, outputVector = null;
        int idPattern = ((RPInputWordSynLayer) inputLayer).getIndexFromWord(word);
        HashMap<String, INDArray> wordSynonymsVectors = ((RPInputWordSynLayer) inputLayer).getVectorSyns(word);
        if(wordSynonymsVectors==null){
            this.numberOriginalVectors++;
            return getCliqueOutput(word);
        }   
        wordSynonymsVectors.put(word, inputWordVector);
        int numLayers = 0;
        // Verificar para numero de layers != 1 e existem outras layers que nao sao differential
        for (RPDenseLayer l : layers) {
            RPDifferentialLayer diffLayer = (RPDifferentialLayer) l;
            for (String s : wordSynonymsVectors.keySet()) {
                if (numLayers == 0) {
                    inputVector = l.getActivationsMult(wordSynonymsVectors.get(s));
                } else {
                    // verificar essa atribuicao
                    inputVector = null;
                }
                diffLayer.memorizeSampleLocalWTA(inputVector, idPattern);
            }
            outputVector = diffLayer.getVectorShortTermMemory(idPattern, thresholdSTM);
            numLayers++;
        }
        return outputVector;
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
