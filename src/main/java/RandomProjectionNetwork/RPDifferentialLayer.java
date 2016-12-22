/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package RandomProjectionNetwork;

import java.util.ArrayList;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author msobroza
 */
public class RPDifferentialLayer extends RPDenseLayer {

    private final ShortTermMemory stm;
    

    public RPDifferentialLayer(int inputDimension, int numClusters, int fanalsPerCluster, float min, float max) {
        super(inputDimension, numClusters, fanalsPerCluster, min, max);
        this.stm=null;
    }

    public RPDifferentialLayer(int inputDimension, int numClusters, int fanalsPerCluster, int numWords, float min, float max) {
        super(inputDimension, numClusters, fanalsPerCluster, min, max);
        this.stm = new ShortTermMemory(numWords, numClusters, fanalsPerCluster);
    }

    public INDArray getVectorDifferentialWTA(INDArray activations, Double threshold, int idPattern) {
        INDArray clusterActivations = activations.reshape(1, numClusters, fanalsPerCluster);
        INDArray indexWinners = Nd4j.argMax(clusterActivations, 2);
        INDArray diff = Nd4j.max(clusterActivations, 2).sub(Nd4j.min(clusterActivations, 2));
        INDArray result = Nd4j.zeros(fanalsPerCluster * numClusters);
        ArrayList<Integer> globalIndexesClique= new ArrayList<>();
        for (int iCluster = 0; iCluster < numClusters; iCluster++) {
            if (diff.getDouble(iCluster) > threshold) {
                int idFanal = iCluster * fanalsPerCluster + indexWinners.getInt(iCluster);
                result.putScalar(idFanal, 1.0);
                globalIndexesClique.add(idFanal);
            }
        }
        this.stm.memActivationsGlobalIndex(idPattern, globalIndexesClique);
        return result;
    }

    public INDArray getVectorDifferentialWTA(INDArray activations, int idPattern) {
        int[] cliqueIndexes = getCliqueIndexesWTA(activations);
        this.stm.memActivationsLocalIndex(idPattern, cliqueIndexes);
        return getVectorFromCliques(cliqueIndexes);
    }

    public INDArray getVectorDifferentialWTA(INDArray activations, Double threshold) {
        INDArray clusterActivations = activations.reshape(1, numClusters, fanalsPerCluster);
        INDArray indexWinners = Nd4j.argMax(clusterActivations, 2);
        INDArray diff = Nd4j.max(clusterActivations, 2).sub(Nd4j.min(clusterActivations, 2));
        INDArray result = Nd4j.zeros(fanalsPerCluster * numClusters);
        for (int iCluster = 0; iCluster < numClusters; iCluster++) {
            if (diff.getDouble(iCluster) > threshold) {
                int idFanal = iCluster * fanalsPerCluster + indexWinners.getInt(iCluster);
                result.putScalar(idFanal, 1.0);
            }
        }
        return result;
    }

    public INDArray getVectorDifferentialWTA(INDArray activations) {
        return getVectorFromCliques(getCliqueIndexesWTA(activations));
    }
}
