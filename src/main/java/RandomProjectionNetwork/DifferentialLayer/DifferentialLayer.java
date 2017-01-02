/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package RandomProjectionNetwork.DifferentialLayer;

import RandomProjectionNetwork.RPDenseLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author msobroza
 */
public class DifferentialLayer extends RPDenseLayer {
    
    public DifferentialLayer(int numWords, int numClusters, int fanalsPerCluster, float min, float max) {
        super(numWords, numClusters, fanalsPerCluster, min, max);
    }
    
    public static INDArray getDiffMaxMin(INDArray activations, int numClusters, int fanalsPerCluster) {
        INDArray clusterActivations = activations.reshape(1, numClusters, fanalsPerCluster);
        return Nd4j.max(clusterActivations, 2).sub(Nd4j.min(clusterActivations, 2));
    }
    
    public static INDArray getDivDiffMax(INDArray activations, int numClusters, int fanalsPerCluster) {
        INDArray clusterActivations = activations.reshape(1, numClusters, fanalsPerCluster);
        return getDiffMaxMin(activations, numClusters, fanalsPerCluster).div(Nd4j.max(clusterActivations, 2));
    }
    
}
