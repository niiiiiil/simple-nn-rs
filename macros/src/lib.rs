extern crate proc_macro;

use proc_macro2::{self, Ident, TokenStream};

use quote::{format_ident, quote};
use syn::{ConstParam, Fields, Generics, ImplGenerics, ItemStruct, TypeGenerics, parse2, parse_macro_input, parse_quote};

/// Derive [`Layers`] with given number of layers for the aimed struct, assuming it is `T`.
/// 
/// A [`Calculation`] impl will also be generated, which has generics same as `T` .
/// 
/// Besides the [`Layers`] trait, the struct `T` will also implement a function
/// `random()` to create an instance with all parameters random value, as well as
/// the `Default` trait to create an instance with all parameters zero value.
/// 
/// The generated struct will have `layer_count + 1` const generics,
/// for the first generic represents the input layer.
/// 
/// # Example
/// 
/// ```
/// #[derive_layers(3)]
/// struct EmampleLayers{}
/// ```
/// 
/// Above will be generated into:
/// 
/// ```
/// struct EmampleLayers<
///     const L0: usize,
///     const L1: usize,
///     const L2: usize,
///     const L3: usize,
/// > {
///     pub layer_1: Layer<L1,L0>,
///     pub layer_2: Layer<L2,L1>,
///     pub layer_3: Layer<L3,L2>,
/// }
/// ```
/// 
/// where `EmampleLayers` impl `Layers<EmampleLayersCal<L0,L1,L2,L3>, L0>`.
/// 
/// so call `forward` on `EmampleLayers` will produce a `EmampleLayersCal<L0,L1,L2,L3>`,
/// which impl `Calculation<EmampleLayers<L0,L1,L2,L3>>`:
/// 
/// ```
/// struct EmampleLayersCal<
///     const L0: usize,
///     const L1: usize,
///     const L2: usize,
///     const L3: usize,
/// >{
///     pub z_1: SVector<f64,L1>,
///     pub a_1: SVector<f64,L1>,
///     pub z_2: SVector<f64,L2>,
///     pub a_2: SVector<f64,L2>,
///     pub z_3: SVector<f64,L3>,
///     pub a_3: SVector<f64,L3>,
/// }
/// ```
/// 
#[proc_macro_attribute]
pub fn derive_layers(layer_count: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(item as syn::ItemStruct);
    let layer_count = parse_macro_input!(
        layer_count as syn::LitInt
    ).base10_parse::<usize>().unwrap();

    let strct = gen_struct(input, layer_count);
    let impl_layers = impl_layers(&strct);
    let impl_random = impl_random(&strct);
    
    (quote! {
        #[derive(Default)]
        #strct
        #impl_random
        #impl_layers
    }).into()
}

/// generate generics params and fields.
fn gen_struct(strct: ItemStruct, layer_count: usize) -> ItemStruct {
    let mut fields = TokenStream::new();  
    let mut generics = quote! {const L0: usize,};
    let mut pre = format_ident!("L0");

    for i in 1..=layer_count {
        let layer = format_ident!("layer_{}", i);
        let cur = format_ident!("L{}", i);
        fields.extend(quote! {pub #layer: Layer<#cur, #pre>,});
        generics.extend(quote! {const #cur: usize, });
        pre = cur;
    }
    
    let vis = &strct.vis;
    let name = &strct.ident;

    let strct = quote! {
        #vis struct #name <#generics> {
            #fields
        }
    };
    
    parse2(strct).unwrap()
}

fn impl_layers(strct: &ItemStruct) -> TokenStream {
    let name = &strct.ident;
    let calc_name = format_ident!("{}Cal", name);
    let generics = &strct.generics;

    let mut const_params = generics.const_params().map(|c| &c.ident);
    let input_size = const_params.next().unwrap();
    let output_size = const_params.last().unwrap();

    let (_, type_generics, _) = generics.split_for_impl();

    let mut impl_generics = quote! {F,};
    for c in generics.const_params() {
        impl_generics.extend(quote!{#c,});
    }

    let [impl_forward, impl_test] = impl_forward(&calc_name, generics.const_params());
    let impl_backward = impl_backward(generics);
    let impl_update = impl_update(generics);

    let mut impletation = quote!{
        impl<#impl_generics> Layers<F, #calc_name #type_generics, #input_size, #output_size> for #name #type_generics
        where
            F: ActivitionFunc
        {
            fn forward(&self, item: &SVector<f64,#input_size>) -> #calc_name #type_generics {
                #impl_forward
            }
            fn backward(&self, input: &SVector<f64,#input_size>, pE_pOut: SVector<f64, #output_size>, calc: #calc_name #type_generics) 
                -> Self {
                #impl_backward
            }
            fn update(&mut self, rate: f64, gradients: impl IntoIterator<Item = Self>) {
                #impl_update
            }
            fn test(&self, item: &SVector<f64, #input_size>) -> SVector<f64, #output_size> {
                #impl_test
                z
            }
        }
    };

    impletation.extend(gen_calc(&calc_name, &strct));

    impletation
}

fn impl_forward<'a>(calc_name: &Ident, iter: impl Iterator<Item = &'a ConstParam>) -> [TokenStream;2] {
    let mut impl_forward = quote! {
        let z_1 = self.layer_1.calc(item);
        let a_1 = z_1.map(F::f);
    };
    let mut impl_test = quote! {
        let mut z = self.layer_1.calc(item);
        z.iter_mut().for_each(|z| *z = F::f(*z));
    };
    let mut calc_fields = quote! {z_1, a_1,};

    // impl forward & update
    for (cur, _) in iter.map(|c| &c.ident).enumerate().skip(2) {
        let pre = cur - 1;
        let a_pre = format_ident!("a_{}", pre);
        let a = format_ident!("a_{}", cur);
        let z = format_ident!("z_{}", cur);
        let layer = format_ident!("layer_{}", cur);
        impl_forward.extend(quote! {
            let #z = self.#layer.calc(&#a_pre);
            let #a = #z.map(F::f);
        });
        calc_fields.extend(quote! {
            #z, #a,
        });
        impl_test.extend(quote! {
            let mut z = self.#layer.calc(&z);
            z.iter_mut().for_each(|z| *z = F::f(*z));
        });
    }
    let impl_forward = quote! {
        #impl_forward
        #calc_name {
            #calc_fields
        }
    };
    [impl_forward, impl_test]
}

fn impl_backward(generics: &Generics) -> TokenStream {

    let iter = generics.params.iter().enumerate().skip(1).rev();
    // let (last,_) = iter.next().unwrap();

    let mut impl_backward = quote! {};
    let mut k = quote! {pE_pOut};
    let mut fields = TokenStream::new();

    for (cur, _) in iter {
        let a = format_ident!("a_{}", cur);
        let a_pre = if cur == 1 { quote!(input) } else {
            let ident = format_ident!("a_{}", cur-1);
            quote! {calc.#ident}
        };
        let layer = format_ident!("layer_{}", cur);
        impl_backward.extend(quote! {
            let delta = calc.#a.zip_map(&(#k), |z, k| { k * F::d_from_y(z) });
            let #layer = Layer {
                w: delta * #a_pre.transpose(),
                b: delta,
            };
        });
        fields.extend(quote!{
            #layer,
        });
        k = quote! {self.#layer.w.transpose() * delta};
    }
    quote! {
        #impl_backward

        Self {#fields}
    }
}

fn impl_update(generics: &Generics) -> TokenStream {
    let mut sum = TokenStream::new();
    let mut update = TokenStream::new();
    for (i,_) in generics.params.iter().enumerate().skip(1) {
        let layer = format_ident!("layer_{}", i);
        sum.extend(quote! {
            f.#layer += g.#layer.as_ref();
        });
        update.extend(quote! {
            f.#layer /= len;
            f.#layer *= rate;
            self.#layer -= f.#layer.as_ref();
        });
    }
    quote! {
        let mut iter = gradients.into_iter().enumerate();
        let (mut len, mut f) = iter.next().unwrap();

        for (i,g) in iter {
            len = i;
            #sum
        }
        let len = (len + 1) as f64;
        #update
    }
}

/// impl layers constructor with random init params
fn impl_random(strct: &ItemStruct) -> TokenStream {
    let name = &strct.ident;
    let (impl_generics, type_generics, _) = strct.generics.split_for_impl();

    let mut random_fields = TokenStream::new();
    for f in strct.fields.iter().map(|f| f.ident.as_ref().unwrap()) {
        random_fields.extend(quote! {
            #f: Layer::random(),
        });
    }

    quote! {
        impl #impl_generics #name #type_generics {
            pub fn random() -> Self {
                Self {
                    #random_fields
                }
            }
        }
    }
}

fn gen_calc(calc_name: &Ident, layers: &ItemStruct) -> TokenStream {
    let mut fields = TokenStream::new();  

    let generics = layers.generics.const_params().skip(1);
    for (i, const_param) in generics.enumerate() {
        let a = format_ident!("a_{}", i+1);
        let z = format_ident!("z_{}", i+1);
        let len = &const_param.ident;
        fields.extend((quote! {
            pub #z: SVector<f64, #len>,
            pub #a: SVector<f64, #len>,
        }).into_iter());
    }

    let (impl_generics,_,_) = layers.generics.split_for_impl();

    let calc: ItemStruct = parse_quote!{
        struct #calc_name #impl_generics {
            #fields
        }
    };

    let __impl = impl_calc(&calc);
    
    quote! {
        #[derive(Clone)]
        #calc
        #__impl
    }
}

fn impl_calc(calc: &ItemStruct) -> TokenStream {
    #![allow(non_snake_case)]

    let calc_name = &calc.ident;

    let (impl_generics, type_generics, _) = &calc.generics.split_for_impl();

    let impl_default = impl_calc_default(calc_name, &calc.fields, impl_generics, type_generics);
    // let impl_basic = impl_calc_basic(calc, impl_generics, type_generics);

    let (last_index, last_const) = calc.generics
            .const_params()
            .map(|c| &c.ident)
            .enumerate()
            .last()
            .unwrap();
    let E = quote! {#last_const};
    let out = format_ident!("a_{}", last_index);

    quote! {
        #impl_default

        impl #impl_generics Calculation<#E> for #calc_name #type_generics {
            fn out(&self) -> &SVector<f64, #E> {
                &self.#out
            }
        }
    }
}

fn impl_calc_default(
    calc_name: &Ident,
    fields: &Fields,
    impl_generics: &ImplGenerics,
    type_generics: &TypeGenerics
) -> TokenStream {
    let mut default_fields = TokenStream::new();

    for field  in fields.iter().map(|f| f.ident.as_ref().unwrap()) {
        default_fields.extend(quote! { #field: SVector::repeat(0f64), });
    }

    quote! {
        impl #impl_generics Default for #calc_name #type_generics {
            fn default() -> Self {
                Self {
                    #default_fields
                }
            }
        }
    }
}
