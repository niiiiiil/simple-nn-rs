
use std::marker::PhantomData;

use itertools::Itertools;
use rand::prelude::SliceRandom;

use crate::{Config, Item, func::*, model::*};

pub struct Trainer<'a, A, C, L, Cal, const I: usize, const O: usize> {
    data: Vec<&'a Item<I, O>>,
    layers: L,
    config: Config<A, C>,
    item_lstn: Option<Box<dyn 'a + Fn(TempModel<L,A,Cal,I,O>, &Config<A,C>, &Item<I,O>, &SVector<f64, O>)>>,
    chunk_lstn: Option<Box<dyn 'a + Fn(TempModel<L,A,Cal,I,O>, &Config<A,C>, &[&Item<I,O>], &[SVector<f64, O>])>>,
    iter_lstn: Option<Box<dyn 'a + Fn(usize, TempModel<L,A,Cal,I,O>, &Config<A,C>)>>,
    _maker: std::marker::PhantomData<Cal>
}

impl<'a, A, C, L, Cal, const I: usize, const O: usize> Trainer<'a, A, C, L, Cal, I, O>
where
    A: ActivitionFunc,
    C: LossFunc,
    L: Layers<A, Cal, I, O>,
    Cal: Calculation<O>
{
    #[inline]
    pub fn new<T>(layers: L, items: T, config: Config<A, C>) -> Self
    where
        T: Iterator<Item = &'a self::Item<I, O>>
    {
        Self {
            data: items.collect(),
            config,
            layers,
            item_lstn: None,
            chunk_lstn: None,
            iter_lstn: None,
            _maker: Default::default()
        }
    }

    pub fn build(mut self) -> Model<L, A, Cal, I, O> {
        #![allow(non_snake_case)]

        let mut rng = rand::thread_rng();
    
        for i in 1..self.config.iter_num+1 {
            self.data.shuffle(&mut rng);
            let chunks = self.data
                .chunks(self.config.batch_size)
                .collect_vec();

            for chunk in chunks {
                let mut outputs = Vec::with_capacity(chunk.len());

                let layers = &mut self.layers;
                let config = &self.config;
                let item_listener = &self.item_lstn;

                let gradients = chunk.iter()
                    .map(|item| {
                        let calc = layers.forward(&item.data);
                        let out = calc.out().to_owned();
                        outputs.push(out);
                        let last_gradient = C::d(&item.label, calc.out());

                        if let Some(f) = item_listener {
                            f(TempModel::new(layers), config, item, &out);
                        }

                        layers.backward(&item.data, last_gradient, calc)
                    }).collect_vec();

                layers.update(self.config.learn_rate, gradients);   

                self.call_chunk_listeners(chunk, &outputs);
            }

            self.call_iter_lstn(i);

            if i % 10_000 == 0 {
                self.config.learn_rate /= 2f64;
            }
        }

        Model::new(self.layers)
    }

    #[inline]
    pub fn after_each_item<F>(&mut self, f: F)
    where
        F: 'a + Fn(TempModel<L,A,Cal,I,O>, &Config<A,C>, &Item<I,O>, &SVector<f64, O>),
    {
        self.item_lstn = Some(Box::new(f));
    }

    // fn call_item_listeners(&self, item: &Item<I,O>, output: &SVector<f64, O>) {
    //     for f in self.item_listener.iter() {
    //         f(&self.layers, &self.config, item, output);
    //     }
    // }

    #[inline]
    pub fn after_each_chunk<F>(&mut self, f: F)
    where
        F: 'a + Fn(TempModel<L,A,Cal,I,O>, &Config<A,C>, &[&Item<I,O>], &[SVector<f64, O>]),
        L: Layers<A,Cal,I,O>,
    {
        self.chunk_lstn = Some(Box::new(f));
    }

    #[inline]
    fn call_chunk_listeners(&self, chunk: &[&Item<I,O>], outputs: &[SVector<f64, O>]) {
        if let Some(f) = &self.chunk_lstn {
            f(TempModel::new(&self.layers), &self.config, chunk, outputs);
        }
    }

    #[inline]
    pub fn after_each_iter<F>(&mut self, f: F)
    where
        F: 'a + Fn(usize, TempModel<L,A,Cal,I,O>, &Config<A,C>),
    {
        self.iter_lstn = Some(Box::new(f));
    }

    #[inline]
    fn call_iter_lstn(&self, iter_num: usize) {
        if let Some(f) = &self.iter_lstn {
            f(iter_num,TempModel::new(&self.layers), &self.config);
        }
    }
}

pub struct TempModel<'a, L, F, C, const I: usize, const O: usize> {
    pub layers: &'a L,
    _maker: PhantomData<(F,C)>,
}

impl<'a, L, F, C, const I: usize, const O: usize> TempModel<'a,L, F, C,I,O> 
where
    L: Layers<F, C, I, O>,
    C: Calculation<O>
{
    fn new(layers: &'a L) -> Self {
        Self {
            layers,
            _maker: Default::default(),
        }
    }

    #[inline]
    pub fn test(&self, item: &SVector<f64, I>) -> SVector<f64,O> {
        self.layers.test(item)
    }
}
