var products = []
$.ajax({
    type: "get",
    url: "./static/js/products_data.csv",
    async: true,
    dataType: 'text',
    success: function(d) {
        products = $.csv2Array(csvd);
    },
    complete: function () {
        // call a function on complete 
    }
});