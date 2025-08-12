from typing import List

from src.validations.spec import LayoutSpec


class DecoderTwoZone:
    """Giải mã genome (hoán vị toàn cục) -> layout 2 vùng (block tủ mát cố định)."""

    def __init__(self, spec: LayoutSpec):
        self.spec = spec

    def decode(self, genome: List[str]) -> List[str]:
        if set(genome) != self.spec.set_all or len(genome) != self.spec.N:
            raise ValueError("Genome không phải hoán vị đúng của all_items.")
        start, end = self.spec.anchor_start, self.spec.anchor_end
        prefix_len = start
        suffix_len = self.spec.N - end

        non_refrig_seq = [g for g in genome if g not in self.spec.set_refrig]
        refrig_seq = [g for g in genome if g in self.spec.set_refrig]

        prefix = non_refrig_seq[:prefix_len]
        suffix = non_refrig_seq[prefix_len : prefix_len + suffix_len]
        return prefix + refrig_seq + suffix


class IdentityDecoder:
    """Không biến đổi, layout = genome."""

    def decode(self, genome: List[str]) -> List[str]:
        return list(genome)
