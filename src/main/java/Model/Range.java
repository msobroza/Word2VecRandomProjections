/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model;

import Model.Embedding.Embedding;
import java.util.HashSet;
import java.util.Random;

/**
 *
 * @author msobroza
 */
public class Range {
    private final int idRange;
    private Double minValue;
    private Double maxValue;
    private final HashSet<Embedding> vectors;
    
    public Range(int idRange, Double minValue, Double maxValue){
        this.idRange = idRange;
        this.maxValue = maxValue;
        this.minValue = minValue;
        this.vectors= new HashSet<>();
    }
    
    public Range(int idRange){
        this.idRange=idRange;
        this.vectors= new HashSet<>();
    }
    
    public Range(int idRange, Double minValue, Double maxValue, HashSet<Embedding> elements){
        this.idRange = idRange;
        this.maxValue = maxValue;
        this.minValue = minValue;
        this.vectors= new HashSet<>(elements);
    }
    
    public void addVector(Embedding e){
        vectors.add(e);
    }
    
    public double getValueInterval(){
        return maxValue-minValue;
    }
    
    public Embedding getRandomEmbedding(){
        //System.out.println("Vectors size in range: "+vectors.size());
        int item = new Random().nextInt(vectors.size());
        int count=0;
        for(Embedding e:vectors){
            if(count==item){
                return e;
            }
            count++;
        }
        return null;
    }
    
    public Double getMaxValue(){
        return this.maxValue;
    }
    
    public HashSet<Embedding> getVectors(){
        return this.vectors;
    }
    
    public boolean isInRange(Embedding e){
        return vectors.contains(e);
    }
    
    public int getIdRange(){
        return this.idRange;
    }
    public Double getMinValue(){
        return this.minValue;
    }
    
}
