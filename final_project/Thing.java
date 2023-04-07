class Poly {
    private class Term {
        private int coef;
        private int expo;
        private Term next;
        public Term(int coef, int expo, Term next) {
            this.coef = coef;
            this.expo = expo;
            this.next = next;
        }
    }
    
    private Term first;
    private Term last;
    public Poly() {
        first = new Term(0, Integer.MAX_VALUE, null);
        last = first;
    }
    public boolean isZero() {
        return this.first.next == null;
    }
    public Poly minus() {
        Poly result = new Poly();
        Term temp = this.first.next;
        while (temp != null) {
            int coef = -temp.coef;
            int expo = temp.expo;
            Term next = temp.next;
            result.last.next = new Term(coef, expo, next);
            result.last = result.last.next;
            temp = temp.next;
        }
        return result;
    }
    public Poly plus(Poly that) {
        Poly result = new Poly();
        Term left = this.first.next;
        Term right = that.first.next;
        while (left != null && right != null) {
            if (left.expo > right.expo) {
                result.last.next = new Term(left.coef, left.expo, left.next);
                result.last = result.last.next;
                left = left.next;
            } else if (right.expo > left.expo) {
                result.last.next = new Term(right.coef, right.expo, right.next);
                result.last = result.last.next;
                right = right.next;
            } else {
                int sum = left.coef + right.coef;
                if (sum != 0) {
                    result.last.next = new Term(sum, left.expo, null);
                    result.last = result.last.next;
                }
                left = left.next;
                right = right.next;
            }
        }
        if (left != null) {
            result.last.next = left;
        } else {
            result.last.next = right;
        }
        return result;
    }
    public Poly plus(int coef, int expo) {
        if (coef == 0) {
            throw new IllegalArgumentException("Coefficient cannot be zero.");
        }
        if (expo < 0) {
            throw new IllegalArgumentException("Exponent cannot be negative.");
        }
        if (last != null && expo >= last.expo) {
            throw new IllegalArgumentException("Exponent cannot be greater than or equal to the exponent of the last term.");
        }
        Term newTerm = new Term(coef, expo,null);
        if (first == null) {
            first = newTerm;
            last = newTerm;
        } else {
            last.next = newTerm;
            last = newTerm;
        }
        return this;
    }
    public String toString() {
        StringBuilder builder = new StringBuilder();
        Term current = first.next;
        if (current == null) {
            builder.append("0");
        } else {
            while (current != null) {
                if (current.coef < 0) {
                    builder.append(current.coef);
                    builder.append('x');
                    appendExpo(builder,current.expo);
                } else {
                    builder.append(current.coef);
                    builder.append('x');
                    appendExpo(builder,current.expo);
                    builder.append(" + ");
                }
                current = current.next;
            }
        }
        return builder.toString();
    }
        private void appendExpo(StringBuilder builder, int expo) 
        { 
        if (expo == 0) 
        { 
            builder.append('⁰'); 
        } 
        else 
        { 
            appendingExpo(builder, expo); 
        } 
        } 
        
        private void appendingExpo(StringBuilder builder, int expo) 
        { 
        if (expo > 0) 
        { 
            appendingExpo(builder, expo / 10); 
            builder.append("⁰¹²³⁴⁵⁶⁷⁸⁹".charAt(expo % 10)); 
        } 
    }
}




class PollyEsther {
    public static void main(String[] args) 
    { 
      Poly p = new Poly().plus(3,5).plus(2,4).plus(2,3).plus(-1,2).plus(5,0); 
      Poly q = new Poly().plus(7,4).plus(1,2).plus(-4,1).plus(-3,0); 
      Poly z = new Poly(); 
   
      System.out.println(p);                 // 3x⁵ + 2x⁴ + 2x³ - 1x² + 5x⁰ 
      System.out.println(q);                 // 7x⁴ + 1x² - 4x¹ - 3x⁰ 
      System.out.println(z);                 // 0 
   
      System.out.println(p.minus());         // -3x⁵ - 2x⁴ - 2x³ + 1x² - 5x⁰ 
      System.out.println(q.minus());         // -7x⁴ - 1x² + 4x¹ + 3x⁰ 
      System.out.println(z.minus());         // 0 
   
      System.out.println(p.plus(q));         // 3x⁵ + 9x⁴ + 2x³ - 4x¹ + 2x⁰ 
      System.out.println(p.plus(z));         // 3x⁵ + 2x⁴ + 2x³ - 1x² + 5x⁰ 
      System.out.println(p.plus(q.minus())); // 3x⁵ - 5x⁴ + 2x³ - 2x² + 4x¹ + 8x⁰ 
    } 
  }
