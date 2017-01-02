/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package RandomProjectionNetwork.STM;

import java.util.ArrayList;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author msobroza
 */
public class ShortTermMemory {

    protected final short[] patternAcc;
    protected final int numClusters;
    protected final int fanalsPerCluster;

    public ShortTermMemory(int numWords, int numClusters, int fanalsPerCluster) {
        patternAcc = new short[numWords];
        for (int i = 0; i < numWords; i++) {
            patternAcc[i] = 0;
        }
        this.numClusters = numClusters;
        this.fanalsPerCluster = fanalsPerCluster;
    }

    protected INDArray getArgMaxIndexes(INDArray activations) {
        INDArray clusterActivations = activations.reshape(1, numClusters, fanalsPerCluster);
        return Nd4j.argMax(clusterActivations, 2);
    }

}
