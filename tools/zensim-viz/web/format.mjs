export function formatNumber(value) {
  if (value == null) return 'unavailable';
  const [mantissa, exponent] = Number(value).toPrecision(4).split('e');
  const compact = mantissa.includes('.') ? mantissa.replace(/0+$/, '').replace(/\.$/, '') : mantissa;
  return compact + (exponent == null ? '' : `e${exponent}`);
}
