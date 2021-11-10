use nn::{Item, model::*};
use rand::prelude::SliceRandom;

use crate::data::IRIS;

pub fn load_data<const I:usize>() -> Vec<Item<I, I>> {
    let mut data = Vec::with_capacity(150);
    for line in IRIS.lines(){
        if line == "" {
            break;
        }
        let segments = parse_line::<I>(&line);
        data.push(segments);
    }
    data.shuffle(&mut rand::thread_rng());
    minmax_scale(data)
}

fn minmax_scale<const I: usize>(mut data: Vec<SVector<f64,I>>) -> Vec<Item<I,I>> {
    let mut min = SVector::<f64,I>::repeat(0f64);
    let mut max = SVector::<f64,I>::repeat(0f64);

    for v in data.iter() {
        for ((a, b), i) in max.iter_mut().zip(min.iter_mut()).zip(v.iter()) {
            if i > a {
                *a = *i;
            } else if i < b {
                *b = *i;
            }
        }
    }
    
    for v in data.iter_mut() {
        for ((a, b), i) in max.iter().zip(min.iter()).zip(v.iter_mut()) {
            *i = (*i - *b) / (a - b)
        }
    }

    data.into_iter().map(|v| {
        Item { data: v.clone(), label: v }
    }).collect()
}

fn parse_line<const I: usize>(line: &str) -> SVector<f64, I> {
    let iter = line.split(',').enumerate().map(|(i, seg)| {
        if i == 4 {
            parse_type(seg)
        } else {
            seg.parse::<f64>().unwrap()
        }
    });
    SVector::from_iterator(iter)
}

fn parse_type(typ: &str) -> f64 {
    match typ {
        "Iris-setosa" => 1.,
        "Iris-versicolor" => 2.,
        "Iris-virginica" => 3.,
        _ => 4.,
    }
}