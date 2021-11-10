//! This example impl an auto-encoder and test using the iris dataset.

extern crate simple_nn as nn;

use nn::{Config, derive_layers, Network, func::{ActivitionFunc, LossFunc, DistanceFunc, Tanh}, model::*};

use tools::*;

mod tools;
mod data;

#[derive_layers(2)]
struct EncoderLayers{}

fn main() {
    let layers = EncoderLayers::<5,3,5>::random();
    let config = Config {
        loss_func: DistanceFunc,
        actvt_func: Tanh,
        learn_rate: 0.1,
        batch_size: 5,
        iter_num: 20_000,
    };

    let net = Network::cfg(layers, config);

    let data = load_data();

    let train = &data[0..30];
    let test = &data[30..];

    let trainner = net.train(train);

    let model = trainner.build();

    // test
    let mut dis_sum = 0.;
    for label in test.iter().map(|i| &i.label) {
        let out = model.test(label);
        dis_sum += DistanceFunc::f(label, &out);
    }

    let avg_cost = dis_sum / test.len() as f64;
    println!("Avg Distance: {:.4}", avg_cost);
}

