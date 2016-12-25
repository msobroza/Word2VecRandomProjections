/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Word2VecParser;

/**
 *
 * @author msobroza
 */
public class Config {
    
    public enum SynonymFile {
        
        IndexWord(0),Word(1),FistSynonym(2);
        private final int index;
        SynonymFile(int index){
            this.index=index;
        }
        
        public int getIndex(){
            return this.index;
        }
        
    }

    public enum SimilarityFunction {

        CosinusDistance(0), SquaredDistance(1), NormL2(2), HammingDistance(3);
        private final int index;

        SimilarityFunction(int index) {
            this.index = index;
        }

        public int getIndex() {
            return this.index;
        }
    }

    public enum BinaryType {

        DoubleType(0), BitType(1);
        private final int index;

        BinaryType(int index) {
            this.index = index;
        }

        public int getIndex() {
            return this.index;
        }
    }

    public enum MethodAnchorPoints {

        Random(0), Sum(1), L1(2), KMeansCentroids(3), RandomBinaryCodes(4), DistributedHadarmardCode(5);
        private final int index;

        MethodAnchorPoints(int index) {
            this.index = index;
        }

        public int getIndex() {
            return this.index;
        }
    }
}
