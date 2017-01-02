/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package RandomProjectionNetwork.DifferentialLayer;

import RandomProjectionNetwork.RPDenseLayer;
import RandomProjectionNetwork.STM.STMHard;
import java.util.ArrayList;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author msobroza
 */
public class RPDifferentialHardLayer extends DifferentialLayer {

    private final STMHard stm;

    public RPDifferentialHardLayer(int inputDimension, int numClusters, int fanalsPerCluster, float min, float max) {
        super(inputDimension, numClusters, fanalsPerCluster, min, max);
        this.stm = null;
    }

    public RPDifferentialHardLayer(int inputDimension, int numClusters, int fanalsPerCluster, int numWords, float min, float max) {
        super(inputDimension, numClusters, fanalsPerCluster, min, max);
        this.stm = new STMHard(numWords, numClusters, fanalsPerCluster);
    }

    public INDArray getBinaryVectorSTM(int idPattern, Double thresholdSTM) {
        return this.getVectorFromIndexes(this.stm.getWTAPattern(idPattern, thresholdSTM));
    }

    public INDArray getBinaryVectorSTM(int idPattern) {
        return this.getVectorFromIndexes(this.stm.getWTAPattern(idPattern));
    }

    // one shot learning
    public void memorizeSampleGlobalWTA(INDArray activationsSample, Double thresholdDifferential, int idPattern) {
        INDArray indexWinners = getArgMaxIndexes(activationsSample);
        INDArray diffDivMax = getDivDiffMax(activationsSample, this.numClusters, this.fanalsPerCluster);
        ArrayList<Integer> globalIndexesClique = new ArrayList<>();
        // Selecting fanals to memorize (using a threshold) (global strategy)
        for (int iCluster = 0; iCluster < numClusters; iCluster++) {
            if (diffDivMax.getDouble(iCluster) >= thresholdDifferential) {
                int idFanal = iCluster * fanalsPerCluster + indexWinners.getInt(iCluster);
                globalIndexesClique.add(idFanal);
            }
        }
        this.stm.memActivationsGlobal(idPattern, globalIndexesClique);
    }

    public void memorizeSampleLocalWTA(INDArray activationsSample, int idPattern) {
        int[] cliqueIndexes = getCliqueLocalWTA(activationsSample);
        this.stm.memActivationsLocal(idPattern, cliqueIndexes);
    }
    
    // Without a short-term memory
    public INDArray getVectorDifferentialWTA(INDArray activations, Double threshold) {
        ArrayList<Integer> resultIndexes = new ArrayList<>();
        INDArray indexWinners = getArgMaxIndexes(activations);
        INDArray diff = getDiffMaxMin(activations, this.numClusters, this.fanalsPerCluster);
        for (int iCluster = 0; iCluster < numClusters; iCluster++) {
            if (diff.getDouble(iCluster) >= threshold) {
                int idFanal = iCluster * fanalsPerCluster + indexWinners.getInt(iCluster);
                resultIndexes.add(idFanal);
            }
        }
        return this.getVectorFromIndexes(resultIndexes);
    }

    public INDArray getVectorLWTA(INDArray activations) {
        return getVectorFromCliques(getCliqueLocalWTA(activations));
    }
}
