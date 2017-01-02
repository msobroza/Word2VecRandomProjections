/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package RandomProjectionNetwork.STM;

import RandomProjectionNetwork.STM.ShortTermMemory;
import static RandomProjectionNetwork.DifferentialLayer.DifferentialLayer.getDiffMaxMin;
import static RandomProjectionNetwork.DifferentialLayer.DifferentialLayer.getDivDiffMax;
import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 *
 * @author msobroza
 */
public class STMSoft extends ShortTermMemory {

    private final INDArray patternSoftActivationAcc;

    public STMSoft(int numWords, int numClusters, int fanalsPerCluster) {
        super(numWords, numClusters, fanalsPerCluster);
        int[] dim = {numWords, numClusters, fanalsPerCluster};
        patternSoftActivationAcc = Nd4j.zeros(dim);
    }

    public INDArray getPatternSoftActivationAcc(int idPattern) {
        return patternSoftActivationAcc.get(NDArrayIndex.point(idPattern - 1), NDArrayIndex.all());
    }

    public void accumulatePatternActivation(int patternId, INDArray activationPattern) {
        INDArray accPattern = patternSoftActivationAcc.get(NDArrayIndex.point(patternId - 1), NDArrayIndex.all());
        accPattern.addi(activationPattern);
    }

    public void memActivationsPattern(int patternId, INDArray activations) {
        // calculate softmax and sum them in the memory
        INDArray softmaxResult;
        INDArray input = activations.reshape(numClusters, fanalsPerCluster);
        softmaxResult = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax", input));
        patternAcc[patternId]++;
        accumulatePatternActivation(patternId, softmaxResult);
    }

    public ArrayList<Integer> getLWTAPattern(int patternId) {
        ArrayList<Integer> result = new ArrayList<>();
        INDArray indexWinners = getArgMaxIndexes(getPatternSoftActivationAcc(patternId));
        for (int iCluster = 0; iCluster < numClusters; iCluster++) {
            result.add(indexWinners.getInt(iCluster) + iCluster * fanalsPerCluster);
        }
        return result;
    }

    public ArrayList<Integer> getWTAPattern(int patternId, Double thresholdSTM) {
        ArrayList<Integer> result = new ArrayList<>();
        INDArray indexWinners = getArgMaxIndexes(getPatternSoftActivationAcc(patternId));
        INDArray diff = getDivDiffMax(getPatternSoftActivationAcc(patternId), numClusters, fanalsPerCluster);
        for (int iCluster = 0; iCluster < numClusters; iCluster++) {
            if (diff.getDouble(iCluster) >= thresholdSTM) {
                result.add(indexWinners.getInt(iCluster) + iCluster * fanalsPerCluster);
            }
        }
        return result;
    }
    
    
}
