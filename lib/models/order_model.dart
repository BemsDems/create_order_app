class Order {
  final int orderId;
  final String status;
  final String? paymentUrl;

  Order({required this.orderId, required this.status, this.paymentUrl});

  factory Order.fromJson(Map<String, dynamic> json) {
    return Order(
      orderId: json['order_id'],
      status: json['status'],
      paymentUrl: json['payment_url'],
    );
  }
}
