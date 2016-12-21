/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package CliqueNetwork;

import Model.Anchor.Anchor;
import Model.Embedding.Embedding;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

/**
 *
 * @author msobroza
 */
public class ClusterClique {

    private final Anchor centroid;
    private final ArrayList<Fanal> fanals;
    private final HashMap<Embedding, Fanal> mapVectorFanal;
    private final int[] scoreFanals;

    public ClusterClique(Anchor centroid, int firstFanalId, int lastFanalId) {
        this.centroid = centroid;
        this.fanals = new ArrayList<>();
        for (int i = firstFanalId; i <= lastFanalId; i++) {
            this.fanals.add(i, new Fanal(i, this));
        }
        scoreFanals = new int[this.fanals.size()];
        this.mapVectorFanal = new HashMap<>();
    }

    public Fanal mapRandomFanal(Embedding e) {
        int i = new Random().nextInt(fanals.size());
        Fanal f = fanals.get(i);
        mapVectorFanal.put(e, f);
        return f;
    }
    
    public void activateFanal(Fanal f, int activationValue){
        scoreFanals[fanals.indexOf(f)]+=activationValue;
    }

    public Anchor getCentroid() {
        return this.centroid;
    }

    public int getNumFanals() {
        return this.fanals.size();
    }
    
    public ArrayList<Fanal> LWsTA(){
        ArrayList<Fanal> result= new ArrayList<>();
        int maxScore=getMaxScore();
        if(maxScore==0)
            return result;
        for(int i=0; i<fanals.size();i++){
            if(scoreFanals[i]==maxScore){
                result.add(fanals.get(i));
            }       
        }
        return result;
    }
    
    private int getMaxScore() {
        int maxValue = Integer.MIN_VALUE;
        for (int i = 0; i < scoreFanals.length; i++) {
            if (maxValue < scoreFanals[i]) {
                maxValue=scoreFanals[i];
            }
        }
        return maxValue;
    }

    public void initScore() {
        for (int i = 0; i < getNumFanals(); i++) {
            scoreFanals[i] = 0;
        }
    }

}
