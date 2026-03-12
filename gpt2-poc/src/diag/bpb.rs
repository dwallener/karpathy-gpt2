pub fn loss_to_bpb(loss: f32) -> f32 {
    loss / std::f32::consts::LN_2
}
