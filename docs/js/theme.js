(function($) {
	'use strict';

	var nav_offset_top = $('header').height() + 50;
	/*-------------------------------------------------------------------------------
	  Navbar 
	-------------------------------------------------------------------------------*/

	//* Navbar Fixed
	function navbarFixed() {
		if ($('.header_area').length) {
			$(window).scroll(function() {
				var scroll = $(window).scrollTop();
				if (scroll >= nav_offset_top) {
					$('.header_area').addClass('navbar_fixed');
				} else {
					$('.header_area').removeClass('navbar_fixed');
				}
			});
		}
	}
	navbarFixed();

	var dropToggle = $('.menu_right > li').has('ul').children('a');
	dropToggle.on('click', function() {
		dropToggle.not(this).closest('li').find('ul').slideUp(200);
		$(this).closest('li').children('ul').slideToggle(200);
		return false;
	});

	$('.toggle_icon').on('click', function() {
		$('body').toggleClass('open');
	});

	$('.side_menu .list.menu_right').mCustomScrollbar({
		theme: 'dark'
	});

	/*----------------------------------------------------*/
	/*  Magnific Pop up js
	/*----------------------------------------------------*/

	// for img popup //
	$('.package-area').magnificPopup({
		delegate: '.img-popup',
		type: 'image',
		gallery: {
			enabled: true
		}
	});

	$('.play-video').magnificPopup({
		disableOn: 700,
		type: 'iframe',
		mainClass: 'mfp-fade',
		removalDelay: 160,
		preloader: false,

		fixedContentPos: false
	});

	/*----------------------------------------------------*/
	/*  Home Slider js
    /*----------------------------------------------------*/
	var swiper = new Swiper('.swiper-container', {
		autoplay: {
			delay: 5000
		},
		speed: 2000,
		loop: true
	});

	/*----------------------------------------------------*/
	/*  Portfolio carousel js
    /*----------------------------------------------------*/

	$('.active-gallery-carousel').owlCarousel({
		items: 2,
		// autoplay: 2500,
		loop: true,
		margin: 30,
		nav: true,
		navText: [ "<img src='img/cprev.png'>", "<img src='img/cnext.png'>" ],
		dots: false,
		responsive: {
			0: {
				items: 1
			},
			420: {
				items: 1
			},
			575: {
				items: 1
			},
			768: {
				items: 2
			},
			1200: {
				items: 2
			},
			1680: {
				items: 2
			}
		}
	});

	/*-------------------------------------------------------------------------------
    Testimonial Slider 
	-------------------------------------------------------------------------------*/
	$('.active_testimonial').owlCarousel({
		items: 1,
		loop: true,
		dots: false,
		autoplay: false,
		nav: true,
		navText: [ "<img src='img/cprev.png'>", "<img src='img/cnext.png'>" ]
	});

	/*----------------------------------------------------*/
	/*  MailChimp Slider
    /*----------------------------------------------------*/
	function mailChimp() {
		$('#mc_embed_signup').find('form').ajaxChimp();
	}
	mailChimp();

	/*----------------------------------------------------*/
	/*  Nice Select
    /*----------------------------------------------------*/

	$('select').niceSelect();

	/*----------------------------------------------------*/
	/*  datepicker 
    /*----------------------------------------------------*/

	/*----------------------------------------------------*/
	/*  Datepicker 
    /*----------------------------------------------------*/
	$('#datepicker').datepicker({
		showOn: 'button',
		buttonImage: 'img/calendar.png',
		buttonImageOnly: true
	});

	$('#datepicker1').datepicker({
		showOn: 'button',
		buttonImage: 'img/calendar.png',
		buttonImageOnly: true
	});

	// Owl Carousel
	// if ($('.owl-banner').length) {
	//     $('.owl-banner').owlCarousel({
	//       items: 1,
	//       loop: true,
	//       margin: 0,
	//       dots: false,
	//     //   autoplay: 2500,
	//       nav: true,
	//       navText: ["<img src='img/next.png'>", "<img src='img/next.png'>"]
	//     });
	//   }

	/*----------------------------------------------------*/
	/*  Google map js
    /*----------------------------------------------------*/

	if ($('#mapBox').length) {
		var $lat = $('#mapBox').data('lat');
		var $lon = $('#mapBox').data('lon');
		var $zoom = $('#mapBox').data('zoom');
		var $marker = $('#mapBox').data('marker');
		var $info = $('#mapBox').data('info');
		var $markerLat = $('#mapBox').data('mlat');
		var $markerLon = $('#mapBox').data('mlon');
		var map = new GMaps({
			el: '#mapBox',
			lat: $lat,
			lng: $lon,
			scrollwheel: false,
			scaleControl: true,
			streetViewControl: false,
			panControl: true,
			disableDoubleClickZoom: true,
			mapTypeControl: false,
			zoom: $zoom,
			styles: [
				{
					featureType: 'water',
					elementType: 'geometry.fill',
					stylers: [
						{
							color: '#dcdfe6'
						}
					]
				},
				{
					featureType: 'transit',
					stylers: [
						{
							color: '#808080'
						},
						{
							visibility: 'off'
						}
					]
				},
				{
					featureType: 'road.highway',
					elementType: 'geometry.stroke',
					stylers: [
						{
							visibility: 'on'
						},
						{
							color: '#dcdfe6'
						}
					]
				},
				{
					featureType: 'road.highway',
					elementType: 'geometry.fill',
					stylers: [
						{
							color: '#ffffff'
						}
					]
				},
				{
					featureType: 'road.local',
					elementType: 'geometry.fill',
					stylers: [
						{
							visibility: 'on'
						},
						{
							color: '#ffffff'
						},
						{
							weight: 1.8
						}
					]
				},
				{
					featureType: 'road.local',
					elementType: 'geometry.stroke',
					stylers: [
						{
							color: '#d7d7d7'
						}
					]
				},
				{
					featureType: 'poi',
					elementType: 'geometry.fill',
					stylers: [
						{
							visibility: 'on'
						},
						{
							color: '#ebebeb'
						}
					]
				},
				{
					featureType: 'administrative',
					elementType: 'geometry',
					stylers: [
						{
							color: '#a7a7a7'
						}
					]
				},
				{
					featureType: 'road.arterial',
					elementType: 'geometry.fill',
					stylers: [
						{
							color: '#ffffff'
						}
					]
				},
				{
					featureType: 'road.arterial',
					elementType: 'geometry.fill',
					stylers: [
						{
							color: '#ffffff'
						}
					]
				},
				{
					featureType: 'landscape',
					elementType: 'geometry.fill',
					stylers: [
						{
							visibility: 'on'
						},
						{
							color: '#efefef'
						}
					]
				},
				{
					featureType: 'road',
					elementType: 'labels.text.fill',
					stylers: [
						{
							color: '#696969'
						}
					]
				},
				{
					featureType: 'administrative',
					elementType: 'labels.text.fill',
					stylers: [
						{
							visibility: 'on'
						},
						{
							color: '#737373'
						}
					]
				},
				{
					featureType: 'poi',
					elementType: 'labels.icon',
					stylers: [
						{
							visibility: 'off'
						}
					]
				},
				{
					featureType: 'poi',
					elementType: 'labels',
					stylers: [
						{
							visibility: 'off'
						}
					]
				},
				{
					featureType: 'road.arterial',
					elementType: 'geometry.stroke',
					stylers: [
						{
							color: '#d6d6d6'
						}
					]
				},
				{
					featureType: 'road',
					elementType: 'labels.icon',
					stylers: [
						{
							visibility: 'off'
						}
					]
				},
				{},
				{
					featureType: 'poi',
					elementType: 'geometry.fill',
					stylers: [
						{
							color: '#dadada'
						}
					]
				}
			]
		});
	}
})(jQuery);
