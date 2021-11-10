//! This module exemplifies what the [`derive_layers`] macro generates.
//! 
//! The module's contents is what the following content will be generated into:
//! 
//! ```
//! #[derive_layers(2)]
//! pub struct EncoderLayers{}
//! ```

extern crate neural_network as nn;

use nn::{func::*, model::*};

#[derive(Default)]
pub struct EncoderLayers<
    const L0: usize,
    const L1: usize,
    const L2: usize
>{
    pub layer_1: Layer<L1,L0>,
    pub layer_2: Layer<L2,L1>,
}

pub struct EncoderLayersCal<
    const L0: usize,
    const L1: usize,
    const L2: usize,
>{
    a_1: SVector<f64,L1>,
    z_1: SVector<f64,L1>,
    a_2: SVector<f64,L2>,
    z_2: SVector<f64,L2>,
}

impl<const L0: usize,const L1: usize,const L2: usize> EncoderLayers<L0,L1,L2> {
    pub fn random() -> Self {
        Self {
            layer_1: Layer::random(),
            layer_2: Layer::random(),
        }
    }
}

impl<F: ActivitionFunc, const L0: usize,const L1: usize,const L2: usize> Layers<F, EncoderLayersCal<L0,L1,L2>,L0,L2> for EncoderLayers<L0,L1,L2> {

    fn forward(&self, item: &SVector<f64,L0>) -> EncoderLayersCal<L0,L1,L2> {
        let z_1 = self.layer_1.calc(item);
        let a_1 = z_1.map(F::f);
        let z_2 = self.layer_2.calc(&a_1);
        let a_2 = z_2.map(F::f);
        EncoderLayersCal {
            z_1, a_1, z_2, a_2,
        }
    }

    fn test(&self, item: &SVector<f64, L0>) -> SVector<f64,L2> {
        let mut z = self.layer_1.calc(item);
        z.iter_mut().for_each(|z| *z = F::f(*z));
        let mut z = self.layer_2.calc(&z);
        z.iter_mut().for_each(|z| *z = F::f(*z));
        z
    }

    #[allow(non_snake_case)]
    fn backward(&self, input: &SVector<f64,L0>, pE_pOut: SVector<f64, L2>, calc: EncoderLayersCal<L0,L1,L2>) -> Self {
        
        let k = &pE_pOut;
        let delta = calc.a_2.zip_map(k, |y, k| { k * F::d_from_y(y) });
        let layer_2 = Layer {
            w: delta * calc.a_1.transpose(),
            b: delta,
        };

        let k = self.layer_2.w.transpose() * delta;
        let delta = calc.a_1.zip_map(&k, |y, k| { k * F::d_from_y(y) });
        let layer_1 = Layer {
            w: delta * input.transpose(),
            b: delta,
        };
        
        Self {
            layer_1,
            layer_2,
        }
    }

    fn update(& mut self, rate: f64, gradients: impl IntoIterator<Item = Self>) {
        let mut iter = gradients.into_iter().enumerate();
        let (mut len, mut f) = iter.next().unwrap();
        
        // sum
        for (i,g) in iter {
            len = i;
            f.layer_1 += &g.layer_1;
            f.layer_2 += &g.layer_2;
        }

        // average
        let len = (len + 1) as f64;
        f.layer_1 /= len;
        f.layer_1 *= rate;
        
        f.layer_2 /= len;
        f.layer_2 *= rate;
        
        self.layer_1 -= &f.layer_1;
        self.layer_2 -= &f.layer_2
    }
}

impl<const L0: usize,const L1: usize,const L2: usize> Calculation<L2> for EncoderLayersCal<L0,L1,L2> {
    fn out(&self) -> &SVector<f64,L2> {
        &self.a_2
    }
}