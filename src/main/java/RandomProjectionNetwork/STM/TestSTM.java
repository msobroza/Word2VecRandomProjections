/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package RandomProjectionNetwork.STM;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author msobroza
 */
public class TestSTM {

    public static void main(String[] args) {
        System.out.println("teste 1");
        STMSoft s = new STMSoft((short)4, 6, 2);
        System.out.println("teste 2");
        INDArray x = Nd4j.rand(6, 2);
        INDArray y = Nd4j.rand(6, 2);
        INDArray clusterActivationsX = x.reshape(1, 6, 2);
        INDArray clusterActivationsY = y.reshape(1, 6, 2);
        System.out.println("Activity x: ");
        System.out.println(clusterActivationsX);
        s.memActivationsPattern(1, x);
        System.out.println("Activity y: ");
        System.out.println(clusterActivationsY);
        System.out.println("-----------");
        s.memActivationsPattern(1, y);
        System.out.println("Result of accumulator: ");
        System.out.println(s.getPatternSoftActivationAcc(1));
    }
}
